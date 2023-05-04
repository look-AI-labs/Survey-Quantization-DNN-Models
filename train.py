import argparse
import os
import time
import socket
import logging
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import models
from models.losses import CrossEntropyLossSoft
from data import get_dataset, get_transform
from optimizer import get_optimizer_config, get_lr_scheduler
from utils import setup_logging,  save_checkpoint #setup_gpus,
from utils import AverageMeter, accuracy

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--results-dir', default='./results', help='results dir')
parser.add_argument('--dataset', default='cifar10', help='dataset name or folder, cifar10, imagenet')
parser.add_argument('--train_split', default='train', help='train split name')
parser.add_argument('--model', default='resnet20q', help='model architecture')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
parser.add_argument('--optimizer', default='sgd', help='optimizer function used, adam, sgd')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate, 0.1, 0.01, 0.001')
parser.add_argument('--lr_decay', default='100,150,180', help='lr decay steps')
parser.add_argument('--weight-decay', default=3e-4, type=float, help='weight decay, 3e-4, 0')
parser.add_argument('--print-freq', '-p', default=20, type=int, help='print frequency')
parser.add_argument('--pretrain', default=None, help='path to pretrained full-precision checkpoint')
parser.add_argument('--resume', default=None, help='path to latest checkpoint')
parser.add_argument('--bit_width_list', default='4,8', help='bit width list')
parser.add_argument('--device', default='gpu', help='cpu or gpu')
args = parser.parse_args()


def main():
    hostname = socket.gethostname()
    if not os.path.exists(args.results_dir) : 
        os.makedirs(args.results_dir)
        print('The result dir is created : ' + args.results_dir) 
    setup_logging(os.path.join(args.results_dir, 'log_{}.txt'.format(hostname)))
    logging.info("running arguments: %s", args)

    #best_gpu = setup_gpus()
    #torch.cuda.set_device(best_gpu)
    #torch.backends.cudnn.benchmark = True
    '''
    loading train data from cifar10,cifar100, imagenet using torchvision
    the origianl RGB data are normolized by transform with mean/std pre-calculated, describled in data.py
    '''
    train_transform = get_transform(args.dataset, 'train')
    train_data = get_dataset(args.dataset, args.train_split, train_transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,  # speeding up data processing by multiple threads.
                                               pin_memory=True)

    val_transform = get_transform(args.dataset, 'val')
    val_data = get_dataset(args.dataset, 'val', val_transform)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    #device = torch.device('cpu')
    '''
    quantization bits : input format like ' --bit_width_list 1,2,4,8,32 ', so bit_width_list = [ 1, 2, 4, 8, 32 ]
    '''
    bit_width_list = list(map(int, args.bit_width_list.split(',')))
    bit_width_list.sort()
    '''
    All supported models saved in dir models, including resnet20q, resnet50q, svhn_quan, and svhnq
    '''
    model = models.__dict__[args.model](bit_width_list, train_data.num_classes)#.to(device) #.cuda()
    if args.device == 'gpu' : model = model.cuda()

    '''
    lr_decay : learning rate decay set " --lr_decay "100,150,180" "  lr_decay =[ 100, 150, 180] 
    optimizer set as adam
    '''
    lr_decay = list(map(int, args.lr_decay.split(',')))
    optimizer = get_optimizer_config(model, args.optimizer, args.lr, args.weight_decay)
    lr_scheduler = None
    best_prec1 = None
    
    # continue training from the last point by saving parameters of the model 
    if args.resume and args.resume != 'None':
        if os.path.isdir(args.resume):
            args.resume = os.path.join(args.resume, 'model_best.pth.tar')
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(best_gpu))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler = get_lr_scheduler(args.optimizer, optimizer, lr_decay, checkpoint['epoch'])
            logging.info("loaded resume checkpoint '%s' (epoch %s)", args.resume, checkpoint['epoch'])
        else:
            raise ValueError('Pretrained model path error!')
    # loading pretrained best parameters into model and continue quantations for the best 
    elif args.pretrain and args.pretrain != 'None':
        if os.path.isdir(args.pretrain):
            args.pretrain = os.path.join(args.pretrain, 'model_best.pth.tar')
        if os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location='cuda:{}'.format(best_gpu))
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("loaded pretrain checkpoint '%s' (epoch %s)", args.pretrain, checkpoint['epoch'])
        else:
            raise ValueError('Pretrained model path error!')
    if lr_scheduler is None:
        lr_scheduler = get_lr_scheduler(args.optimizer, optimizer, lr_decay)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    if args.device == 'gpu' : 
        criterion      = nn.CrossEntropyLoss().cuda()
        criterion_soft = CrossEntropyLossSoft().cuda()
    else :
        criterion      = nn.CrossEntropyLoss()
        criterion_soft = CrossEntropyLossSoft()

    sum_writer = SummaryWriter(args.results_dir + '/summary')

    for epoch in range(args.start_epoch, args.epochs):
        ''' model.train() set model in training mode, ie., update batchnorm layers
        see, https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.train 
        '''
        model.train()
        train_loss, train_prec1, train_prec5 = forward(train_loader, model, criterion, criterion_soft, epoch, True,
                                                       optimizer, sum_writer)
        model.eval()
        val_loss, val_prec1, val_prec5 = forward(val_loader, model, criterion, criterion_soft, epoch, False)

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()

        if best_prec1 is None:
            is_best = True
            best_prec1 = val_prec1[-1]
        else:
            is_best = val_prec1[-1] > best_prec1
            best_prec1 = max(val_prec1[-1], best_prec1)
        if False : save_checkpoint(
            {
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict()
            },
            is_best,
            path=args.results_dir + '/ckpt')

        if False and sum_writer is not None:
            sum_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)
            for bw, tl, tp1, tp5, vl, vp1, vp5 in zip(bit_width_list, train_loss, train_prec1, train_prec5, val_loss,
                                                      val_prec1, val_prec5):
                sum_writer.add_scalar('train_loss_{}'.format(bw), tl, global_step=epoch)
                sum_writer.add_scalar('train_prec_1_{}'.format(bw), tp1, global_step=epoch)
                sum_writer.add_scalar('train_prec_5_{}'.format(bw), tp5, global_step=epoch)
                sum_writer.add_scalar('val_loss_{}'.format(bw), vl, global_step=epoch)
                sum_writer.add_scalar('val_prec_1_{}'.format(bw), vp1, global_step=epoch)
                sum_writer.add_scalar('val_prec_5_{}'.format(bw), vp5, global_step=epoch)
        logging.info('Epoch {}: \ntrain loss {:.2f}, train prec1 {:.2f}, train prec5 {:.2f}\n'
                     '  val loss {:.2f},   val prec1 {:.2f},   val prec5 {:.2f}'.format(
                         epoch, train_loss[-1], train_prec1[-1], train_prec5[-1], val_loss[-1], val_prec1[-1],
                         val_prec5[-1]))


def forward(data_loader, model, criterion, criterion_soft, epoch, training=True, optimizer=None, sum_writer=None):
    # bit_width_list = [ 1, 2, 4, 8, 32] from input args --bit_width '1,2,4,8,32'
    bit_width_list = list(map(int, args.bit_width_list.split(',')))
    bit_width_list.sort()

    ''' Averagetmeter is to save and computate the accumulated sum and accumulated average. i.e., all individual diff between pred/target acculated/computed
    see, https://github.com/pytorch/examples/blob/main/imagenet/main.py
    '''
    losses = [AverageMeter() for _ in bit_width_list]
    top1 = [AverageMeter() for _ in bit_width_list]
    top5 = [AverageMeter() for _ in bit_width_list]

    for i, (input, target) in enumerate(data_loader):
        if not training:
            with torch.no_grad():
                if args.device == 'gpu' :  
                    input = input.cuda()
                    target = target.cuda(non_blocking=True)

                for bw, am_l, am_t1, am_t5 in zip(bit_width_list, losses, top1, top5):
                    model.apply(lambda m: setattr(m, 'wbit', bw))
                    model.apply(lambda m: setattr(m, 'abit', bw))
                    output = model(input)
                    loss = criterion(output, target)

                    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                    am_l.update(loss.item(), input.size(0))
                    am_t1.update(prec1.item(), input.size(0))
                    am_t5.update(prec5.item(), input.size(0))
        else:
            if args.device == 'gpu' :  
                input = input.cuda()
                target = target.cuda(non_blocking=True)

            optimizer.zero_grad()

            # train full-precision supervisor
            model.apply(lambda m: setattr(m, 'wbit', bit_width_list[-1]))
            model.apply(lambda m: setattr(m, 'abit', bit_width_list[-1]))
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses[-1].update(loss.item(), input.size(0))
            top1[-1].update(prec1.item(), input.size(0))
            top5[-1].update(prec5.item(), input.size(0))

            # train less-bit-wdith models
            target_soft = torch.nn.functional.softmax(output.detach(), dim=1)
            for bw, am_l, am_t1, am_t5 in zip(bit_width_list[:-1][::-1], losses[:-1][::-1], top1[:-1][::-1],
                                              top5[:-1][::-1]):
                model.apply(lambda m: setattr(m, 'wbit', bw))
                model.apply(lambda m: setattr(m, 'abit', bw))
                output = model(input)
                # hard cross entropy
                # loss = criterion(output, target)
                # soft cross entropy
                loss = criterion_soft(output, target_soft)
                loss.backward()
                # recursive supervision
                target_soft = torch.nn.functional.softmax(output.detach(), dim=1)

                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                am_l.update(loss.item(), input.size(0))
                am_t1.update(prec1.item(), input.size(0))
                am_t5.update(prec5.item(), input.size(0))
            optimizer.step()

            if i % args.print_freq == 0:
                logging.info('epoch {0}, iter {1}/{2}, bit_width_max loss {3:.2f}, prec1 {4:.2f}, prec5 {5:.2f}'.format(
                    epoch, i, len(data_loader), losses[-1].val, top1[-1].val, top5[-1].val))

    return [_.avg for _ in losses], [_.avg for _ in top1], [_.avg for _ in top5]


if __name__ == '__main__':
    main()
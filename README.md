# Survey-Quantization-DNN-Models
This is to quantize FP 8/4/2bits for CNN models. More codes on the way. 


### Dependency 
* Python 3.7
* PyTorch 1.1.0
* torchvision 0.2.1
* [tensorboardX](https://github.com/lanpa/tensorboardX)


### Train
#### Resnet-20 Models on CIFAR10
Just Run the script
 ```
 python train.py 
 ```
Note that config is already set by default but can be changed in the command line, e.g., 

 ```
 python train.py --device 'gpu' --epochs 100 --model 'resnet20q' 
 ```
 
 ### Reference 
 
 * [1] [Any-Precision Deep Neural Networks](https://arxiv.org/abs/1911.07346).

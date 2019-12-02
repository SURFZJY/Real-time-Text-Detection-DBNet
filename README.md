# Real-time-Text-Detection

PyTorch re-implementation of [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)

<img src="https://github.com/SURFZJY/Real-time-Text-Detection/blob/master/demo/dbnet.png" alt="contour" >

### Difference between thesis and this implementation

1. Use dice loss instead of BCE(binary cross-entropy) loss.

2. Use normal convolution rather than deformable convolution in the backbone network.

3. The architecture of the backbone network is a simple FPN.

4. Have not implement OHEM.

5. The ground truth of the threshold map is constant 1 rather than 'the distance to the closest segment'.

### Introduction

thanks to these project:

- https://github.com/WenmuZhou/PAN.pytorch

The features are summarized blow:

+ Use **resnet18/resnet50/shufflenetV2** as backbone.  

### Contents

1. [Installation](#installation)
2. [Download](#download)
3. [Train](#train)
4. [Predict](#predict)
5. [Eval](#eval)
6. [Demo](#demo)


### Installation

1. pytorch 1.1.0
 
### Download

1. ShuffleNet_V2 Models trained on ICDAR 2013+2015 (training set) 
https://pan.baidu.com/s/19XYTkQcOFZEPLbo5BalV3w

### Train

1. modify genText.py to generate txt list file for training/testing data

2. modify config.json

3. run 

```python
python train.py
```

### Predict

1. run 
```python
python predict.py
```

### Eval

run
```python
python eval.py
```

### Examples

<img src="https://github.com/SURFZJY/Real-time-Text-Detection/blob/master/demo/contour.png" width = "200" height = "300" alt="contour" >

<img src="https://github.com/SURFZJY/Real-time-Text-Detection/blob/master/demo/bbox.png" width = "200" height = "300" alt="bbox" >

### Todo

- [ ] MobileNet backbone

- [ ] Deformable convolution

- [ ] tensorboard support

- [ ] FPN --> Architecture in the thesis

- [ ] Dice Loss --> BCE Loss

- [ ] threshold map gt use 1 --> threshold map gt use distance （Use 1 will accelerate the label generation）

- [ ] OHEM 

- [ ] OpenCV_DNN inference API for CPU machine

- [ ] Caffe version (for deploying with MNN/NCNN)

- [ ] ICDAR13 / ICDAR15 / CTW1500 / MLT2017 / Total-Text 

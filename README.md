# MobileNetv2-SSDLite
Caffe implementation of SSD detection on MobileNetv2, converted from tensorflow.

### Prerequisites
Tensorflow and Caffe version [SSD](https://github.com/weiliu89/caffe) is properly installed on your computer.

### Usage
0. Firstly you should download the original model from [tensorflow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Or use the following command.

```shell
cd MobileNetv2-SSDLite/ssdlite/ 
wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz 
tar -zvxf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
```

1. Use gen_model.py to generate the train.prototxt and deploy.prototxt (or use the default prototxt).

   CLASS_NUM = number of classes + 1. (1 presents background)
```python
python gen_model.py -s deploy -c CLASS_NUM >deploy.prototxt
```
2. Use dump_tensorflow_weights.py to dump the weights of conv layer and batchnorm layer.
3. Use load_caffe_weights.py to load the dumped weights to deploy.caffemodel.
4. Use the code in src to accelerate your training if you have a cudnn7, or add "engine: CAFFE" to your depthwise convolution layer to solve the memory issue.
5. The original tensorflow model is trained on MSCOCO dataset, maybe you need deploy.caffemodel for VOC dataset, use coco2voc.py to get deploy_voc.caffemodel.

### Train your own dataset
1. Generate the trainval_lmdb and test_lmdb from your dataset.
2. Write a labelmap.prototxt
3. Use gen_model.py to generate some prototxt files, replace the "CLASS_NUM" with class number of your own dataset and background. eg. VOC dataset has 20 classes, now CLASS_NUM should be 21. Offset problem is the pad diff between TensorFlow and caffe, so add "--tfpad" can solve this issue.
```shell
python gen_model.py -s train -c CLASS_NUM --tfpad >train.prototxt
python gen_model.py -s test -c CLASS_NUM --tfpad >test.prototxt
python gen_model.py -s deploy -c CLASS_NUM --tfpad >deploy.prototxt
```
4. Copy coco/solver_train.prototxt and coco/train.sh to your project and start training.

### Note
There are some differences between caffe and tensorflow implementation:
1. The padding method 'SAME' in tensorflow sometimes use the [0, 0, 1, 1] paddings, means that top=0, left=0, bottom=1, right=1 padding. In caffe, there is no parameters can be used to do that kind of padding.
2. MobileNet on Tensorflow use ReLU6 layer y = min(max(x, 0), 6), but caffe has no ReLU6 layer. Replace ReLU6 with ReLU cause a bit accuracy drop in ssd-mobilenetv2, but very large drop in ssdlite-mobilenetv2. There is a ReLU6 layer implementation in my fork of [ssd](https://github.com/chuanqi305/ssd).



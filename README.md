# HOI

#### Configs
* Ubuntu 18.04.3
* python 3.7.5
* gcc 7.4.0
* NVIDIA Driver 396.54
* CUDA 9.1
* conda create -n HOI python=3.7.5
* conda install pytorch=0.4.1 cuda91 -c pytorch
* conda install torchvision
* Make packages in models folder(sh make.sh)
* mkdir checkpoints
* mkdir datasets
* mkdir pretrained_models

#### Datasets
* VOC PASCAL 2007 and 2012: https://pjreddie.com/projects/pascal-voc-dataset-mirror/
* IMAGENET VID 2015: http://bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php
* Daly: http://thoth.inrialpes.fr/daly/

#### Pretrained models:
* VGG16 pretrained cafee: https://drive.google.com/drive/folders/10ZWulcPJmC9jHDUIMyKfldeH_a6UIEvn?usp=sharing

#### Reference repos
* https://github.com/ruotianluo/pytorch-faster-rcnn
* https://github.com/jwyang/faster-rcnn.pytorch
* https://github.com/chenyuntc/simple-faster-rcnn-pytorch

#### TODO: Improve documentation

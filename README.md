# Multi-Camera Traffic Analysys Project 
By Kent Ngo, Ethan Paek, Jackson Tseng, Tyler Niiyama, Justin Liu, Spencer Tsang

Advisor David Anastasiu

### Requirements
* python3.6

Setting up the NX Jetsons Xavier
```
sudo apt-get update
sudo apt-get upgrade  
sudo apt-get install python3-dev python3-pip  
sudo apt-get install llvm-8*  
export LLVMCONFIG=/usr/bin/llvm-config-8  
sudo pip3 install -U pip  
sudo pip3 install Cython==0.29.22  
sudo pip3 install numba==0.46.0  
sudo pip3 install llvmlite==0.32.1  
sudo pip3 install scikit-learn==0.21.2  
sudo pip3 install tqdm==4.33.0
sudo pip3 install ffmpeg==1.4  
sudo pip3 install ffmpeg-python==0.2.0
sudo pip3 install pandas==0.22.0 
sudo pip3 install onnx==1.4.1
sudo pip3 install onnxruntime==1.7.0
sudo pip3 install numpy==1.18.1  
sudo pip3 install setuptools==53.0.0  
sudo pip3 install testresources==2.0.1

```

### Installing Pytorch 1.7.0
We built Pytorch 1.7.0 from Nvidia's website (https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048).
```
cd

export OPENBLAS_CORETYPE=ARMV8

wget https://nvidia.box.com/shared/static/cs3xn3td6sfgtene6jdvsxlr366m2dhq.whl -O torch-1.7.0-cp36-cp36m-linux_aarch64.whl

sudo apt-get install python3-pip libopenblas-base libopenmpi-dev

sudo pip3 install torch-1.7.0-cp36-cp36m-linux_aarch64.whl

```

### Installing torchvision 0.8.0
We downloaded torchvision 0.8.0 from the github source (https://github.com/pytorch/vision torchvision).

```
cd

sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev

git clone --branch v0.8.0 https://github.com/pytorch/vision torchvision

export BUILD_VERSION=0.8.0

cd torchvision/

sudo python3 setup.py install

```

### Building Our Pre-trained Model

We would be following JKJung's github (https://github.com/jkjung-avt/tensorrt_demos) and using his yolov3_288.trt pretrained object detection models. Note: The NX Jetson Xavier have already installed Tensorrt. If the user's local machine does not have TensorRT, it can be installed locally following NVIDIA's TensorRT webpage (https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

#### Installing Pycuda

```
cd

mkdir project

cd project/

git clone https://github.com/jkjung-avt/jetson_nano.git

cd jetson_nano

./install_basics.sh

source ${HOME}/.bashrc

sudo apt install build-essential 

sudo apt install make 

sudo apt install cmake 

sudo apt install cmake-curses-gui

sudo apt install git 

sudo apt install g++ 

sudo apt install pkg-config 

sudo apt install curl 

sudo apt install libfreetype6-dev 

sudo apt install libcanberra-gtk-module 

sudo apt install libcanberra-gtk3-module

./install_protobuf-3.8.0.sh
```

#### Installing Tensorflow
```

```

#### Building Yolov3 Model
```
cd

cd project/

git clone https://github.com/jkjung-avt/tensorrt\_demos.git

cd tensorrt\_demos/ssd

./install.sh

./build\_engines.sh

cd ../plugins

vi Makefile 

# Make changes in the Makefile and locate the environment variables \$TENSORRT\_INCS
# and \$TENSORRT\_LIBS. Change the values according to the path where the TensorRT dynamic
# libraries are stored in. The \$TENSORRT\_INCS path should have "trtexec"
# dynamic library and the \$TENSORRT\_LIBS path variable should have libnvinfer.so, libnvparsers.so,
# and other TensorRT dynamic libraries. Since our NX Jetson Xavier have already installed Tensorrt, 
# our environment variables like this:
TENSORRT\_INCS=-I"/usr/src/tensorrt/bin"
TENSORRT\_LIBS=-L"/usr/lib/aarch64-linux-gnu"

# Move libyolo\_layer.so into our Real-Time Multi-Camera Traffic project under folder plugins/.

make

Check if pycuda has successfully been installed.

cd tensorrt\_demos/yolo

./download\_yolo.sh

python3 yolo\_to\_onnx.py -m yolov3-288

python3 onnx\_to\_tensorrt.py -m yolov3-288

# Move the yolov3-288.trt output into our Real-Time Multi-Camera Traffic project under folder yolo/.

```
### Inference

Here are the steps to reproduce our results:

1. Download the corresponding model file [best.pt](https://drive.google.com/open?id=1BaCOU5ABwFMSjbc8frrAIpC6Dp0zTQJz) and put it in the folder `weights`.
2. Make sure the raw video files and required txt files are in the folder `data/Dataset_A`.
3. Run `inference.py` to get separate result files in the folder `output` for all 31 videos.
4. Run `result.py` to combine all 31 csv files and get the single submission file `track1.txt`.

```
mkdir weights
mkdir output
python3 inference.py 1 31
python3 result.py
```

We use YOLOv3+sort to detect and track vehicles. To count the movement, we use a detection line (detection line) for each movement by annotating the provided training videos (Data set A), as defined in `get_lines.py`. If a vehicle passes the detection line, the count of the corresponding movement will increase by 1 after a short pre-defined delay calculated based on the training data.

### Training the detector
We use yolov3 as our detector, which is initialized by the public COCO pre-trained model and fine-tuned with some annotated frames from the training set (which will be described later). The corresponding files are in the folder `yolov3_pytorch`. Below are the steps to train the detector.

1. Make sure the following files are placed correctly.
	* The training images (extracted from the raw training videos) in `data/images/`
	* The annotation text files in `yolov3_pytorch/data/labels/`
	* `object.data` and `object.names` in `yolov3_pytorch/data`, which describe the input data and output classes
2. Downdload the official coco pretrained model [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) from [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/) and put it in `yolov3_pytorch/weights`.
3. Use the following train command to finetine the pretrained model. The `train_best.pt` file is the final model.
```
cd yolov3_pytorch
unzip data/labels.zip
python3 train.py --data data/object.data --cfg cfg/yolov3.cfg --epochs 200
```
Some of the utility scripts are borrowed from https://github.com/ultralytics/yolov3.

##### Annotation Data

We selected 5 videos from the provided training videos (Data set A), including `cam3.mp4, cam5.mp4, cam7.mp4, cam8.mp4, cam20.mp4`. A subset of 3835 frames was extracted from these videos for manual annotation.

You can use the following command to extract frames directly from the videos.
```
ffmpeg -i cam_x.mp4 -r 1 -f image2 yolov3_pytorch/data/images/%06d.jpg
```
The extracted frames should be put in the folder `yolov3_pytorch/data/images`.


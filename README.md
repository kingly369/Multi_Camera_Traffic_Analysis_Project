# Multi-Camera Traffic Analysis Project 
By Kent Ngo, Ethan Paek, Jackson Tseng, Tyler Niiyama, Justin Liu, Spencer Tsang

Advisor David Anastasiu


### Introduction
For our project, we wanted help improve efficiency of traffic lights and collect real-time traffic analysis, such as tracking and counting cars. We created a multiple IOT-based network using NX Jetson Xaviers and utilize them to do vehicle tracking and counting to collect information about an intersection. As a result, civil engineers can use these information to do some analyzing, such as determining when is an intersection the most busiest and think of ideas of how to improve traffic efficiency. Although there are already vehicle tracking and counting algorithms, we wanted to design an algorithm that would improve the tracking and counting of these algorithms by using multiple IOTs. The problem is that there would be lots of times where a camera won't have a great view of the vehicles that they're tracking of or even lose track of the cars due obstruction, such as other cars, that may give inaccurate data about the traffic. Therefore, by having a multi-IOT camera network in the same intersection, we'll have multiple point of views to cover this dilemma. If one of the cameras lose track of a car, we can use another camera that can still track the same vehicle and tell the previous camera that the car is still there in the intersection. We'll assign them the same tracking ID to tell that they're both tracking the same vehicle among the Jetsons.

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
sudo apt install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
                      
sudo pip3 install -U future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures pybind11

sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow==1.15.2
```

#### Building Yolov3 Model
```
cd

cd project/

git clone https://github.com/jkjung-avt/tensorrt_demos.git

cd tensorrt_demos/ssd

./install.sh

./build_engines.sh

cd ../plugins

vi Makefile 

# Make changes in the Makefile and locate the environment variables $TENSORRT_INCS
# and $TENSORRT_LIBS. Change the values according to the path where the TensorRT dynamic
# libraries are stored in. The $TENSORRT_INCS path should have "trtexec"
# dynamic library and the $TENSORRT_LIBS path variable should have libnvinfer.so, libnvparsers.so,
# and other TensorRT dynamic libraries. Since our NX Jetson Xavier have already installed Tensorrt, 
# our environment variables like this:
TENSORRT_INCS=-I"/usr/src/tensorrt/bin"
TENSORRT_LIBS=-L"/usr/lib/aarch64-linux-gnu"

# Move libyolo_layer.so into our Real-Time Multi-Camera Traffic project under folder plugins/.

make

Check if pycuda has successfully been installed.

cd tensorrt_demos/yolo

./download_yolo.sh

python3 yolo_to_onnx.py -m yolov3-288

python3 onnx_to_tensorrt.py -m yolov3-288

# Move the yolov3-288.trt output into our Real-Time Multi-Camera Traffic project under folder yolo/.

```

### Setting up RabbitMQ

#### Installing RabbitMQ

This tutorial assumes that the user is installing this on Ubuntu and other Linux-based architectures.
```
sudo apt-get install apt-transport-https
sudo apt-key adv --keyserver "hkps://keys.openpgp.org" --recv-keys "0x0A9AF2115F4687BD29803A206B73A36E6026DFCA"
sudo apt-key adv --keyserver "keyserver.ubuntu.com" --recv-keys "F77F1EDA57EBB1CC"
curl -1sLf 'https://packagecloud.io/rabbitmq/rabbitmq-server/gpgkey' | sudo apt-key add -
sudo tee /etc/apt/sources.list.d/rabbitmq.list \textless\textless EOF
deb http://ppa.launchpad.net/rabbitmq/rabbitmq-erlang/ubuntu bionic main
deb-src http://ppa.launchpad.net/rabbitmq/rabbitmq-erlang/ubuntu bionic main
deb https://packagecloud.io/rabbitmq/rabbitmq-server/ubuntu/ bionic main
deb-src https://packagecloud.io/rabbitmq/rabbitmq-server/ubuntu/ bionic main
EOF
sudo apt-get update -y
sudo apt-get install -y erlang-base \
    erlang-asn1 erlang-crypto erlang-eldap erlang-ftp erlang-inets \
    erlang-mnesia erlang-os-mon erlang-parsetools erlang-public-key \
    erlang-runtime-tools erlang-snmp erlang-ssl \
    erlang-syntax-tools erlang-tftp erlang-tools erlang-xmerl
sudo apt-get install rabbitmq-server -y --fix-missing
pip install pika
service rabbitmq-server start
Enter admin password for your device.
``` 
The server should be running on your local device. One can then use the example files on RabbitMQ's tutorials on the official website. This, however, does not allow other devices on a local network to connect to the server. This topic will be covered next.

#### Connecting Other Devices onto the RabbitMQ Local Network
This creates an authorized user with the username 'qa1' and password 'yourPassword'.
```
sudo rabbitmqctl add_user qa1 yourPassword
```

This creates a virtual host with the name 'virtualHost1'.

```
sudo rabbitmqctl add_vhost virtualHost1
```

This allows user 'qa1' permission to read and write within 'virtualHost1'.
```
sudo rabbitmqctl set_permissions -p virtualHost1 qa1 ".*" ".*" ".*"
```

Note: The username, password, and name of the virtual host will all be used in the functions PlainCredentials, BlockingConnection, and ConnectionParameters within the pika library.


### Inference

Here are the steps run our program:
1. We need to have 2 NX Jetson Xaviers setup with the following instructions above and cloned the github to both Jetsons. One of the Jetson would be the helper Jetson while the other Jetson would be the receiver who would be getting helped.
2. Create a directory called 'output', which would hold the counting analysis. The counting analysis may not be working properly, and need some time to fix during this time.
3. Make sure your your RabbitMQ server is running. If not, run: service rabbitmq-server start
4. Make sure you successfully setup building our pre-trained yolov3-288 model.
5. Make sure the video files are in the folder `data/Track3/`. So far we have only annotated one intersection called s04.3.
6. Run `inference_helper.py s04.3 17b` on one of the jetson and run `inference_receiver.py s04.3 15b`. It'll output the bounding boxes video and the tracking ID.

We've created hardcoded annotations on the videos to determine region of interest where the cars pass certain lines in order to count vehicles in `get_lines_track3.py`. We also have other hardcoded annotations to determine the region of where both of the interserction is the area where both of the cameras/jetsons can see. For example, we have camera 17b can see the same intersection as camera 15b. We collected the video cameras from AI_City_Challenge 2020 Track 3 (source: https://www.aicitychallenge.org/2020-challenge-tracks) and selected cameras in the same intersection. We finally organized these cameras by intersection in folders, such as s04.3. Only cameras annotated are s04.3/15a (helper) with s04.3/16a (receiver) and s04.3/17b (helper) with s04.3/15b.

If you would like to add other video cameras or intersections, make sure it follows the structure of data/Track3/{intersection}/{video_camera}.mp4


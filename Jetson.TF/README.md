# smart home lobe.ai
smart home example based on lobe.ai image classification on Nvidia Jetson Nano and shelly relays.

## setup

Jetson nano image is already preinstalled with OpenCV, the only requierment which needed to get install is Tensoflow version 1.15.3 in order to run Lobe exported TF model, this process can take ~40min

### install Tensorflor

the following commands are based on the a guide by Nvidia https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html

```
sudo apt-get update
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo apt-get install python3-pip
pip3 install -U pip testresources setuptools==49.6.0
pip3 install -U numpy==1.16.1 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 'tensorflow==1.15.3'
sudo reboot
```

### setup classifier 
```
git clone https://github.com/guybartal/smart-home-lobe.git
cd smart-home-lobe
sudo pip3 install -r requirements.txt
mkdir /data/models
mkdir /data/output
```
upload your expored lobe tf model folder into /data/models

## run
```
python3 classifier.py --model_dir /data/models/tf/hands --output_dir /data/output
```

## dockerized version (Alternative option)
incase you want to run the classifier inside docker, use this command:

```
docker run -it --name lobe --net=host --runtime nvidia  -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-5.0   -v /tmp/argus_socket:/tmp/argus_socket -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu -v /usr/lib/python3.6/dist-packages/cv2:/usr/lib/python3.6/dist-packages/cv2 -v /home/nvidia/ds_workspace:/home/mounted_workspace -v /data/models:/data/models  --device /dev/video0:/dev/video0 -v /tmp/.X11-unix/:/tmp/.X11-unix guybartal/smarthomelobe:v0.3 /bin/bash

cd /app
python3 classifier.py --model_dir /data/models/replace_with_your_model_dir --output_dir /data/output
```
TODO: run classifier inside dockerfile
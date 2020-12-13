# smart home lobe.ai
smart home example based on lobe.ai image classification on Nvidia Jetson Nano and shelly relays.

## setup

Jetson nano image is already preinstalled with OpenCV and gstreamer, the only requierment which needed to get install is onnxruntime to run Lobe exported TF model

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
python3 classifier.py --model_dir /data/models/onnx/hands --output_dir /data/output
```

## dockerized version (Alternative option)
incase you want to run the classifier inside docker, use this command:

```
docker run -it --name lobe --net=host --runtime nvidia  -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-5.0   -v /tmp/argus_socket:/tmp/argus_socket -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu -v /usr/lib/python3.6/dist-packages/cv2:/usr/lib/python3.6/dist-packages/cv2 -v /home/nvidia/ds_workspace:/home/mounted_workspace -v /data/models:/data/models  --device /dev/video0:/dev/video0 -v /tmp/.X11-unix/:/tmp/.X11-unix guybartal/smarthomelobe:v0.3 /bin/bash

cd /app
python3 classifier.py --model_dir /data/models/replace_with_your_model_dir --output_dir /data/output
```
TODO: run classifier inside dockerfile
# smart-home-lobe
smart home example based on lobe.ai image classification and shelly relays

# setup
pip3 install -r requirements.txt

# run
python3 classifier.py

# other options
```
python3 classifier.py -h
usage: classifier.py [-h] [--width WIDTH] [--height HEIGHT]
                     [--persist PERSIST] [--model_path MODEL_PATH]
                     [--post_url POST_URL]

Image classification of video stream from pi camera with lobe.ai model

optional arguments:
  -h, --help            show this help message and exit
  --width WIDTH         image width, default value is 640
  --height HEIGHT       image height, default value is 480
  --persist PERSIST     persist captured frames for model retrating, default
                        value is False
  --model_path MODEL_PATH
                        lobe.ai model path, default value is ./model
  --post_url POST_URL   url to post the inference result to, leave empty if
                        not requeired, defalt is empty
```

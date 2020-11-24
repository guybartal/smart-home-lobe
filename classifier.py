#import Pi GPIO library button class
from picamera import PiCamera
from lobe import ImageModel
from PIL import Image
import numpy as np
import io
import time
import argparse
import requests
import json
import logging

def post_to_url(post_url, result):
  payload = {"Prediction" : result.prediction}
  labels = {}
  for label, prop in result.labels:
    labels.update({label : f"{prop*100}%"})
  payload['Labels'] = labels
  requests.post(post_url, data = json.dumps(payload), headers={"Content-Type": "application/json"})
  
def extract_frame(stream, width, height):
  stream.truncate()
  stream.seek(0)
  stream_value = stream.getvalue()
    
  image = Image.frombuffer('RGB', (width, height), stream_value)
  return image
    

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Image classification of video stream from pi camera with lobe.ai model')
  parser.add_argument('--width', type=int, dest='width', default=640, help='image width, default value is 640', required=False)
  parser.add_argument('--height', type=int, dest='height', default=480, help='image height, default value is 480', required=False)
  parser.add_argument('--persist', type=bool, dest='persist', default=False, help='persist captured frames for model retrating, default value is False', required=False)
  parser.add_argument('--model_path', type=str, dest='model_path', default='./model', help='lobe.ai model path, default value is ./model', required=False)
  parser.add_argument('--post_url', type=str, dest='post_url', default='', help='url to post the inference result to, leave empty if not requeired, defalt is empty', required=False)
  args = parser.parse_args()
  
  logging.basicConfig(format='%(asctime)s - %(levelname)s:%(message)s', level=logging.INFO)
  
  logging.info("loding camera...")
  camera = PiCamera()
  #camera.rotation = 180
  #camera.framerate = 1

  # Load Lobe TF model
  logging.info("loding model...")
  model = ImageModel.load(args.model_path)

  # Main Function
  logging.info("staring capture and predict loop...")
  i = 0
  stream = io.BytesIO() 
  time_stamp = time.time()
  for frame in camera.capture_continuous(stream, format='rgb', use_video_port=True, resize=(args.width, args.height)):                                            
    image = extract_frame(stream, args.width, args.height)  
    
    result = model.predict(image)
  
    if (args.persist):
      image.save(f"image{i:0>5}_{result.prediction}.jpg")
  
    if (args.post_url!=''):
      post_to_url(args.post_url, result)
    
    next_time_stamp = time.time()    
    total_time = next_time_stamp - time_stamp
    logging.info(f"prediction {i}: {result.prediction} total time for frame {total_time:.2f}")    
    time_stamp = next_time_stamp
    i = i+1

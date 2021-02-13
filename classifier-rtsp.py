#  -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation.  All rights reserved.
#  -------------------------------------------------------------
"""
Skeleton code showing how to load and run the TensorFlow SavedModel export package from Lobe.
"""
import argparse
from publisher import Publisher
import os
import json
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import time

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..")  # default assume that our export is in this file's parent directory

class Model(object):
    def __init__(self, model_dir=MODEL_DIR):
        # make sure our exported SavedModel folder exists
        model_path = os.path.realpath(model_dir)
        if not os.path.exists(model_path):
            raise ValueError(f"Exported model folder doesn't exist {model_dir}")
        self.model_path = model_path

        # load our signature json file, this shows us the model inputs and outputs
        # you should open this file and take a look at the inputs/outputs to see their data types, shapes, and names
        with open(os.path.join(model_path, "signature.json"), "r") as f:
            self.signature = json.load(f)
        self.inputs = self.signature.get("inputs")
        self.outputs = self.signature.get("outputs")

        # placeholder for the tensorflow session
        self.session = None
        self.publisher = Publisher()

    def load(self):
        self.cleanup()
        # create a new tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.session = tf.compat.v1.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options = gpu_options))
        # load our model into the session
        tf.compat.v1.saved_model.loader.load(sess=self.session, tags=self.signature.get("tags"), export_dir=self.model_path)

    def predict(self, image: Image.Image):
        # load the model if we don't have a session
        if self.session is None:
            self.load()
        # get the image width and height
        width, height = image.size
        # center crop image (you can substitute any other method to make a square image, such as just resizing or padding edges with 0)
        if width != height:
            square_size = min(width, height)
            left = (width - square_size) / 2
            top = (height - square_size) / 2
            right = (width + square_size) / 2
            bottom = (height + square_size) / 2
            # Crop the center of the image
            image = image.crop((left, top, right, bottom))
        # now the image is square, resize it to be the right shape for the model input
        if "Image" not in self.inputs:
            raise ValueError("Couldn't find Image in model inputs - please report issue to Lobe!")
        input_width, input_height = self.inputs["Image"]["shape"][1:3]
        if image.width != input_width or image.height != input_height:
            image = image.resize((input_width, input_height))
        # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
        image = np.asarray(image) / 255.0
        # create the feed dictionary that is the input to the model
        # first, add our image to the dictionary (comes from our signature.json file)
        feed_dict = {self.inputs["Image"]["name"]: [image]}

        # list the outputs we want from the model -- these come from our signature.json file
        # since we are using dictionaries that could have different orders, make tuples of (key, name) to keep track for putting
        # the results back together in a dictionary
        fetches = [(key, output["name"]) for key, output in self.outputs.items()]

        # run the model! there will be as many outputs from session.run as you have in the fetches list
        outputs = self.session.run(fetches=[name for _, name in fetches], feed_dict=feed_dict)
        # do a bit of postprocessing
        results = {}
        # since we actually ran on a batch of size 1, index out the items from the returned numpy arrays
        for i, (key, _) in enumerate(fetches):
            val = outputs[i].tolist()[0]
            if isinstance(val, bytes):
                val = val.decode()
            results[key] = val
        return results

    def publish(self, msg):
        self.publisher.publish(msg)

    def cleanup(self):
        # close our tensorflow session if one exists
        if self.session is not None:
            self.session.close()
            self.session = None

    def __del__(self):
        self.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict labels from csi camera.")
    parser.add_argument("--rtsp", type=str, dest='rtsp', default='rtsp://guy:qazwsx@192.168.5.242:554//h264Preview_01_main', help="your camera rtsp address.")
    parser.add_argument("--model_dir", type=str, dest='model_dir', default='/data/models/tf/hands', help="your model directory.")
    parser.add_argument("--output_dir", type=str, dest='output_dir', default='/data/output', help="your model directory.")

    args = parser.parse_args()
    model = Model(args.model_dir)
    print("Loading Model")
    model.load()

    print("Starting Video Capture")
    #unmark to capture USB camera
    #videoCapture = cv2.VideoCapture(0)
    videoCapture = cv2.VideoCapture(args.rtsp)

    i = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    timeA = time.time()
    while True:

        ret,image =videoCapture.read()
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_RGB)

        outputs = model.predict(image_pil)
        print(f"Predicted: {outputs}")
        model.publish(outputs)

        if (i%10==0):
            cv2.putText(image,str(outputs),(20,20), font, .5,(255,255,255),2,cv2.LINE_AA)
            file_name = f'{args.output_dir}/frame{i}_{outputs["Prediction"]}.jpg'
            print (f'storing file {file_name}')
            cv2.imwrite(file_name,image)
        i+=1
        timeB= time.time()
        print("Elapsed Time Per Frame: {} Microsec".format(timeB-timeA) )
        timeA=timeB
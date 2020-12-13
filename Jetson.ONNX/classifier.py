#  -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation.  All rights reserved.
#  -------------------------------------------------------------
"""
Skeleton code showing how to load and run the TensorFlow SavedModel export package from Lobe.
"""
import argparse
import os
import json
import onnxruntime as rt

from PIL import Image
import numpy as np
import cv2
import time

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def get_model_and_sig(model_dir):
    """Method to get name of model file. Assumes model is in the parent directory for script."""
    with open(os.path.join(model_dir, "../signature.json"), "r") as f:
        signature = json.load(f)
    
    model_file = os.path.join(model_dir, "../" + signature.get("filename"))
    print (model_file)
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Model file does not exist")
    return model_file, signature

def load_model(model_file):
    """Load the model from path to model file"""
    # Load ONNX model as session.
    return rt.InferenceSession(path_or_bytes=model_file)

def get_prediction(image, session, signature):
    """
    Predict with the ONNX session!
    """
    # get the signature for model inputs and outputs
    signature_inputs = signature.get("inputs")
    signature_outputs = signature.get("outputs")

    if "Image" not in signature_inputs:
        raise ValueError("ONNX model doesn't have 'Image' input! Check signature.json, and please report issue to Lobe.")

    # process image to be compatible with the model
    img = process_image(image, signature_inputs.get("Image").get("shape"))

    # run the model!
    fetches = [(key, value.get("name")) for key, value in signature_outputs.items()]
    # make the image a batch of 1
    feed = {signature_inputs.get("Image").get("name"): [img]}
    outputs = session.run(output_names=[name for (_, name) in fetches], input_feed=feed)
    # un-batch since we ran an image with batch size of 1,
    # convert to normal python types with tolist(), and convert any byte strings to normal strings with .decode()
    results = {}
    for i, (key, _) in enumerate(fetches):
        val = outputs[i].tolist()[0]
        if isinstance(val, bytes):
            val = val.decode()
        results[key] = val

    return results

def process_image(image, input_shape):
    """
    Given a PIL Image, center square crop and resize to fit the expected model input, and convert from [0,255] to [0,1] values.
    """
    width, height = image.size
    # ensure image type is compatible with model and convert if not
    if image.mode != "RGB":
        image = image.convert("RGB")
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
    input_width, input_height = input_shape[1:3]
    if image.width != input_width or image.height != input_height:
        image = image.resize((input_width, input_height))

    # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
    image = np.asarray(image) / 255.0
    # format input as model expects
    return image.astype(np.float32)

def arg_parse():
    parser = argparse.ArgumentParser(description="Predict labels from csi camera.")
    parser.add_argument("--model_dir", type=str, dest='model_dir', default='/data/models/onnx/hands', help="your model directory.")
    parser.add_argument("--output_dir", type=str, dest='output_dir', default='/data/output', help="your model directory.")

    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    print(f"Loading Model from {args.model_dir}")
    model_file, signature = get_model_and_sig(args.model_dir)
    session = load_model(model_file)

    print("Starting Video Capture")  
    #for usb camera unmark this line and mark gstreamer
    #videoCapture = cv2.VideoCapture(0)
    print(gstreamer_pipeline(flip_method=0))
    videoCapture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    i = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    timeA = time.time()
    while True:
        ret,image = videoCapture.read()
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_RGB)

        prediction = get_prediction(image_pil, session, signature)
        print(f"Predicted: {prediction}")

        if (i%30==0):
            cv2.putText(image,str(prediction),(20,20), font, .5,(255,255,255),2,cv2.LINE_AA)
            print (f'storing file {args.output_dir}/frame{i}.jpg')
            cv2.imwrite(f'{args.output_dir}/frame{i}.jpg',image)
        i+=1
        timeB= time.time()
        print("Elapsed Time Per Frame: {} Microsec".format(timeB-timeA) )
        timeA=timeB
    
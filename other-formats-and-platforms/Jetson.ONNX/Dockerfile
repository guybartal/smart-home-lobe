FROM nvcr.io/nvidia/deepstream-l4t:5.0.1-20.09-iot
WORKDIR /app

RUN apt-get update -y
RUN pip3 install opencv-python

COPY ./requirements.txt /app
RUN pip3 install -r requirements.txt
COPY ./classifier.py /app
RUN python3 classifier.py --model_dir /data/models/onnx/hands --output_dir /data/output
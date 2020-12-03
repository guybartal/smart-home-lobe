FROM nvcr.io/nvidia/deepstream-l4t:5.0.1-20.09-iot

RUN apt-get update -y
RUN apt-get install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran python3-pip
RUN pip3 install --no-cache-dir -U pip testresources setuptools==49.6.0
RUN pip3 install --no-cache-dir -U numpy==1.16.1 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
RUN pip3 install --no-cache-dir --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 'tensorflow==1.15.3'
RUN pip3 install opencv-python
#RUN pip3 install git+https://github.com/onnx/tensorflow-onnx

WORKDIR /app
COPY ./requirements.txt /app
RUN pip3 install -r requirements.txt
COPY ./classifier.py /app


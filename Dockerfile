# For more information, please refer to https://aka.ms/vscode-docker-python
FROM public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-pytorch-cuda-full:v1.5.0

USER root

RUN apt-get update && apt-get install -y python3-opencv wget g++

WORKDIR /wd

RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
RUN pip install 'git+https://github.com/cocodataset/panopticapi.git'

COPY requirements.txt /wd
RUN pip install -r requirements.txt

COPY weed_semseg.py /wd

COPY X_Decoder /wd/X_Decoder


RUN chmod -R 777 /wd

USER jovyan


ENTRYPOINT ["python", "-u", "weed_semseg.py"]

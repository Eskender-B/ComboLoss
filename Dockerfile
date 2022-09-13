FROM python:3.7

WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:/app"

COPY ./requirements.txt /tmp/requirements.txt

RUN apt-get update \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/requirements.txt \
    && pip cache purge

RUN pip install fastapi
RUN pip install uvicorn

RUN pip install torch==1.8.2+cpu torchvision==0.9.2+cpu torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

RUN pip install gdown

RUN mkdir /data
RUN gdown https://drive.google.com/uc?id=1O9T41BJglXBWlm6nJCuHIGzl3iSDvBdc -O /data/ComboNet_SCUTFBP5500.pth

RUN pip install pytorchcv

# copy project
COPY . /app

EXPOSE 8000

ENTRYPOINT ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

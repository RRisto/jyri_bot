FROM python:3.6

MAINTAINER Risto Hinno <risto.hinno@gmail.com>

RUN apt-get update && \
    apt-get install -y nano &&\
    apt-get install -y python3-pip

ENV PATH /opt/conda/bin:$PATH

RUN wget https://download.pytorch.org/whl/cpu/torch-1.0.1-cp36-cp36m-linux_x86_64.whl
RUN pip install torch-1.0.1-cp36-cp36m-linux_x86_64.whl
RUN rm torch-1.0.1-cp36-cp36m-linux_x86_64.whl

WORKDIR /workspace
RUN chmod -R a+w /workspace

ENV LANG C.UTF-8

COPY requirements.txt /opt
RUN pip install -r /opt/requirements.txt

RUN python -m spacy download en

RUN mkdir -p /opt/bot

WORKDIR /opt/bot
ADD . /opt/bot
ADD . .

EXPOSE 5000 5001

RUN apt-get clean
RUN apt-get update

EXPOSE 2222 5000 5001

CMD ["python", "run.py"]
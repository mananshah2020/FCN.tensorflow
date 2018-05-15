FROM python:2
ENV PYTHONUNBUFFERED 1
RUN mkdir /fcn
WORKDIR /fcn
RUN apt-get update
RUN apt-get install -y vim
ADD requirements.txt /fcn/
RUN pip install -r requirements.txt
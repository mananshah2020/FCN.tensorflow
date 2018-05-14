FROM python:2
ENV PYTHONUNBUFFERED 1
RUN mkdir /fcn
WORKDIR /fcn
ADD requirements.txt /fcn/
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install -y vim
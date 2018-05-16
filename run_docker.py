import ConfigParser
import os
import docker

config = ConfigParser.ConfigParser()
config.read("settings.ini")

data_dir = config.get('Docker Settings', 'data_dir')
model_dir = config.get('Docker Settings', 'data_dir')
port = config.getint('Docker Settings', 'tensorboard_port')

client = docker.from_env()

print client.containers.run(
    'fcn',
    volumes={data_dir: {'bind': '/dataset', 'mode': 'ro'}, model_dir: {'bind': '/logs', 'mode': 'rw'}},
    ports={6006:port},
    command='python FCN_wrapper.py',
    detach=False,
    auto_remove=True
)
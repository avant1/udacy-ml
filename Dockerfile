FROM python:3.6.8-stretch

COPY requirements.txt requirements.txt

#looks like installing from sources cannot be repeated using requirements file
RUN pip install git+https://github.com/Theano/Theano.git#rel-1.0.4

RUN pip install -r requirements.txt

WORKDIR /opt/app

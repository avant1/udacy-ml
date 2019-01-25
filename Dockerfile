FROM python:3.6.8-stretch

RUN pip install numpy \
            plotly matplotlib \
            scipy scikit-learn \
            nltk \
            pandas numexpr bottleneck \
            git+https://github.com/Theano/Theano.git#egg=Theano \
            keras tensorflow

WORKDIR /opt/app

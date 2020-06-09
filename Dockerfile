FROM continuumio/anaconda3:2020.02

WORKDIR /analysis
COPY requirements.txt .
RUN pip install -U pip && \
  pip install -r requirements.txt -U

FROM python:3.11.0-slim
RUN apt-get update
RUN apt-get install git -y
RUN apt install build-essential -y
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
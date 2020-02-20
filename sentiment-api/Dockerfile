FROM python:3.7-buster

WORKDIR /root/sentiment

COPY . /root/sentiment

EXPOSE 8000:8000

RUN pip install -r requirements.txt && python train.py

CMD ["python", "main.py"]

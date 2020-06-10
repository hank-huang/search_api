FROM python:3.7.0
WORKDIR /api
ADD . /api
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python","api.py"]

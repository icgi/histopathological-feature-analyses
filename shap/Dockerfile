FROM python:3.10-slim

WORKDIR /shap

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt          

RUN apt update && apt -y --no-install-recommends install vim

CMD bash

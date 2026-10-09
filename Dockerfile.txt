
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data.csv .
COPY train_and_predict.py .

RUN mkdir -p /app/output

CMD ["python", "train_and_predict.py"]

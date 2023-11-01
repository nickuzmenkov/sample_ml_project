FROM python:3.11.5

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
ENTRYPOINT ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "app:app"]

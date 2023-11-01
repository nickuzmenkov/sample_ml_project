FROM python:3.11.5

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
ENTRYPOINT ["streamlit", "run", "--server.fileWatcherType", "none", "app.py"]

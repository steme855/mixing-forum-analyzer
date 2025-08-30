
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt &&     python -m spacy download de_core_news_sm || true

COPY . /app
EXPOSE 8080
CMD ["streamlit", "run", "app/app.py", "--server.port", "8080", "--server.headless", "true"]

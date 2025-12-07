FROM python:3.11

RUN apt-get update && apt-get install -y \
    sqlite3 \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY stock_pipeline.py .
COPY sp500_10yr_range_tickers.csv .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-c", "from stock_pipeline import run_sp500_pipeline; run_sp500_pipeline()"]
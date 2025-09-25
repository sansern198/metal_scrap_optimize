FROM python:3.11-slim

# ติดตั้ง build deps สำหรับ opencv และ numpy ให้ smooth
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ด backend
COPY app /app/app

# โฟลเดอร์ที่เราจะเมานต์โปรเจกต์จริงเข้ามา
RUN mkdir -p /app/work/data

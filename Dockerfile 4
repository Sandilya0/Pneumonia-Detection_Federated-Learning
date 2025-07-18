FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ✅ Make start.sh executable
RUN chmod +x start.sh

CMD ["bash", "start.sh"]

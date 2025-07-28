FROM python:3.10-slim

WORKDIR /app

# Copy code and wheels
COPY linux_wheels ./wheels
COPY . .

# Install dependencies offline from wheels directory
RUN pip install --no-index --find-links=./wheels -r requirements.txt

CMD ["python", "main.py"]
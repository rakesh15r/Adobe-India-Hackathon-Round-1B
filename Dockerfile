FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and downloaded wheels
COPY requirements.txt .
COPY downloaded_packages/ ./downloaded_packages/

# Install pip + dependencies from local wheels only
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-index --find-links=downloaded_packages -r requirements.txt

# Copy source code and other files
COPY . .

# Set PYTHONPATH so modules in src/ are importable
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Run your main script
CMD ["python", "main.py"]

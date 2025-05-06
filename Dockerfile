# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy all project files
COPY . .

# Make folders if not present
RUN mkdir -p data/raw data/processed models logs

# Expose Streamlit port
EXPOSE 8501

# Copy and make our entrypoint executable
# (This assumes you've added entrypoint.sh next to main.py.)
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Use the entrypoint script; default to running the full pipeline
ENTRYPOINT ["./entrypoint.sh"]
CMD ["all"]

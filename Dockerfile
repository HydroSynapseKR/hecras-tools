FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (for geopandas & rasterio)
RUN apt-get update && apt-get install -y \
    g++ \
    gcc \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["python"]

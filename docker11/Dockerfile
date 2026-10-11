FROM jupyter/minimal-notebook:lab-4.0.5

# 드라이버가 CUDA 12.0까지 지원하므로 cu118 빌드 사용
RUN pip install --no-cache-dir numpy torch --index-url https://download.pytorch.org/whl/cu118
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install matplotlib pandas ultralytics
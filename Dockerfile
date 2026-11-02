FROM python:3.12-slim

# OpenCV needs these shared libraries even in headless use.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# On Linux, PyPI's torch wheel bundles CUDA (~2 GB). Install the CPU build first; the pinned
# "torch==2.5.1" in requirements.txt is then already satisfied by 2.5.1+cpu.
RUN pip install --no-cache-dir torch==2.5.1+cpu torchvision==0.20.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python scripts/download_models.py

# Example:
#   docker build -t sleep-detector .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/refs:/app/refs" -v "$PWD/outputs:/app/outputs" \
#       sleep-detector --video data/videos/clip1.mp4 --refs refs/ --out outputs/clip1
ENTRYPOINT ["python", "run.py"]

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3 \
    python3-pip

RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 
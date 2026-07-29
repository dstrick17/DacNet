FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1
ENV WANDB_MODE=offline
WORKDIR /workspace/DacNet

# Install system deps for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . /workspace/DacNet

EXPOSE 8501

CMD ["/bin/bash"]

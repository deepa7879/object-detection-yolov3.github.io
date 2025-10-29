# === Stage 1: Base Image ===
FROM python:3.10-slim

# === Stage 2: Install system dependencies (for OpenCV) ===
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# === Stage 3: Set Working Directory ===
WORKDIR /app

# === Stage 4: Copy Files ===
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# === Stage 5: Create Folders for Results ===
RUN mkdir -p results

# === Stage 6: Expose Port and Run App ===
EXPOSE 5000
CMD ["python", "app.py"]

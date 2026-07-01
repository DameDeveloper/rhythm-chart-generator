FROM python:3.12-slim

# ffmpeg is required for audio decoding/conversion (yt-dlp + wav conversion).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Transient audio/song files live here. On most container hosts this is
# ephemeral, which is fine — jobs are meant to be short-lived. Mount a volume
# if you want playback/regeneration to survive restarts.
RUN mkdir -p /app/downloads

# Hosts (Render, Railway, Cloudflare Containers, Fly, ...) inject $PORT.
ENV PORT=8000
EXPOSE 8000

# Single worker: job files are stored on local disk, so regenerate/audio must
# hit the same process. Scale vertically (CPU) rather than to many instances.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

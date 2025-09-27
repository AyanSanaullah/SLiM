# Multi-stage Dockerfile for Google ADK Agents
FROM --platform=linux/amd64 python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY config/ ./config/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY adk_agents/ ./adk_agents/
COPY tools/ ./tools/
COPY app.py .

# Create necessary directories
RUN mkdir -p \
    data/user_data \
    models/user_models \
    logs/user_logs \
    backups/user_models \
    exports/user_models

# Set permissions
RUN chmod -R 755 /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Expose port
EXPOSE 8080

# Default command
CMD ["python", "app.py"]

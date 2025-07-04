# Multi-stage build for Autonomous Trading System
FROM python:3.13-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    cmake \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.13-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash trader

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/trader/.local

# Copy application code
COPY --chown=trader:trader . .

# Create necessary directories
RUN mkdir -p logs data backups && \
    chown -R trader:trader logs data backups

# Switch to non-root user
USER trader

# Add user's Python packages to PATH
ENV PATH=/home/trader/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose monitoring port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Default command
CMD ["python", "-u", "autonomous_trading/main_trading_system.py", "autonomous_trading/config/trading_config.json"]
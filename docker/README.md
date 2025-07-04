# 🐳 Docker Files

Docker configurations for containerized deployment.

## Files
- `Dockerfile` - Main Docker image
- `Dockerfile.multiarch` - Multi-architecture build
- `Dockerfile.node` - Node.js variant
- `docker-compose.yml` - Compose configuration

## Usage
```bash
# Build image
docker build -f docker/Dockerfile -t trading-bot .

# Run with compose
docker-compose -f docker/docker-compose.yml up
```

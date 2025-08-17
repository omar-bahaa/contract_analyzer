# Deployment Guide for Islamic Contract RAG System

## Hosting Options for Free Prototype

### 1. Hugging Face Spaces (Recommended for Streamlit)

**Pros:**
- Free hosting for Streamlit apps
- Good GPU support for ML models
- Easy deployment with Git
- Built-in domain and SSL

**Cons:**
- Limited to Streamlit frontend only
- Backend would need separate hosting

**Setup:**
1. Create account on huggingface.co
2. Create new Space with Streamlit template
3. Upload frontend code and requirements
4. Add backend URL configuration

### 2. Railway (Full Stack)

**Pros:**
- Free tier with 512MB RAM
- Supports both frontend and backend
- PostgreSQL database included
- Automatic HTTPS

**Cons:**
- Limited resources on free tier
- May need optimization for memory usage

**Setup:**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### 3. Render (Full Stack)

**Pros:**
- Free tier with 512MB RAM
- Supports Python/Node.js
- PostgreSQL database
- Custom domains

**Cons:**
- Services sleep after 15 minutes of inactivity
- Limited build minutes

### 4. Google Cloud Run (Scalable)

**Pros:**
- Pay-per-use pricing
- Excellent for containerized apps
- Auto-scaling
- Free tier includes 2 million requests

**Setup:**
```bash
# Build Docker image
docker build -t islamic-contract-rag .

# Deploy to Cloud Run
gcloud run deploy --image gcr.io/PROJECT-ID/islamic-contract-rag
```

### 5. Streamlit Community Cloud

**Pros:**
- Free hosting specifically for Streamlit
- Direct GitHub integration
- Easy deployment

**Cons:**
- Frontend only (need separate backend)
- Limited resources

## Deployment Configurations

### Option 1: Split Deployment (Recommended)

**Backend:** Railway/Render
**Frontend:** Streamlit Community Cloud/Hugging Face Spaces

**Benefits:**
- Leverage best free resources for each component
- Better separation of concerns
- Easier scaling

### Option 2: Containerized Deployment

Create Docker containers for easy deployment across platforms.

**Dockerfile for Backend:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ ./backend/
COPY data/ ./data/

EXPOSE 8000
CMD ["python", "backend/main.py"]
```

**Dockerfile for Frontend:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install streamlit

COPY frontend/ ./frontend/

EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Option 3: Single Server Deployment

For development or small-scale use, deploy everything on one server.

**DigitalOcean Droplet:**
- $5/month basic droplet
- 1GB RAM, 25GB SSD
- Suitable for prototype

**AWS EC2 Free Tier:**
- t2.micro instance
- 1GB RAM, 8GB storage
- 750 hours/month free

## Optimization for Free Hosting

### Memory Optimization:
1. Use lighter ML models
2. Implement model caching
3. Reduce chunk sizes in configuration
4. Use CPU-only versions of libraries

### Performance Optimization:
1. Enable compression in FastAPI
2. Implement response caching
3. Use async operations where possible
4. Optimize database queries

### Cost Optimization:
1. Use free vector databases (ChromaDB)
2. Implement request rate limiting
3. Add request caching
4. Use efficient data storage formats

## Production Considerations

### Security:
- Add authentication and authorization
- Implement API rate limiting
- Use HTTPS everywhere
- Validate all inputs

### Scalability:
- Implement horizontal scaling
- Use load balancers
- Add monitoring and logging
- Use CDN for static assets

### Reliability:
- Add health checks
- Implement circuit breakers
- Use database backups
- Add error monitoring

## Specific Deployment Instructions

### Deploying to Hugging Face Spaces:

1. **Create requirements.txt for frontend:**
```txt
streamlit==1.29.0
requests==2.31.0
```

2. **Create app.py in root:**
```python
import sys
sys.path.append('./frontend')
from frontend.app import main

if __name__ == "__main__":
    main()
```

3. **Create Space on Hugging Face:**
- Go to huggingface.co/spaces
- Create new Space with Streamlit
- Upload files via Git or web interface

### Deploying to Railway:

1. **Create railway.json:**
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "restartPolicyType": "ON_FAILURE",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100
  }
}
```

2. **Environment Variables:**
```
PYTHONPATH=/app/backend
PORT=8000
```

### Deploying to Render:

1. **Create render.yaml:**
```yaml
services:
  - type: web
    name: islamic-contract-rag-backend
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python backend/main.py"
    
  - type: web
    name: islamic-contract-rag-frontend
    env: python
    buildCommand: "pip install streamlit"
    startCommand: "streamlit run frontend/app.py --server.port $PORT"
```

## Monitoring and Maintenance

### Free Monitoring Tools:
1. **UptimeRobot:** Free website monitoring
2. **Google Analytics:** User behavior tracking
3. **Sentry:** Error tracking (free tier)
4. **LogRocket:** Session replay (free tier)

### Backup Strategy:
1. Regular database exports
2. Code versioning with Git
3. Configuration backups
4. Vector database snapshots

## Troubleshooting Common Issues

### Memory Issues:
- Reduce model sizes
- Implement lazy loading
- Use disk-based caching
- Optimize chunk sizes

### Performance Issues:
- Add response caching
- Optimize database queries
- Use async operations
- Implement pagination

### Deployment Issues:
- Check logs for errors
- Verify environment variables
- Test locally first
- Use health check endpoints

This deployment guide provides multiple options for hosting the Islamic Contract RAG system for free, with scalability paths for future growth.

#!/bin/bash
#!/bin/bash

echo "🚀 Deploying Math Agent Application"
echo "=================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Please create a .env file with the required environment variables."
    echo "See .env.example for reference."
    exit 1
fi

# Load environment variables
export $(cat .env | xargs)

# Check required environment variables
required_vars=("GROQ_API_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Required environment variable $var is not set!"
        exit 1
    fi
done

echo "✅ Environment variables validated"

# Create necessary directories
mkdir -p data/raw data/processed logs

# Check if raw data exists
if [ ! "$(ls -A data/raw/*.json 2>/dev/null)" ]; then
    echo "⚠️  Warning: No JSON files found in data/raw/"
    echo "Please place the DeepMind math dataset JSON files in data/raw/ directory"
    echo "The application will create a sample dataset if none is found."
fi

# Build and start services
echo "🔨 Building and starting services..."
docker-compose -f docker/docker-compose.yml up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🏥 Checking service health..."
backend_health=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v1/health)

if [ "$backend_health" -eq 200 ]; then
    echo "✅ Backend service is healthy"
else
    echo "❌ Backend service health check failed"
    echo "Checking logs..."
    docker-compose -f docker/docker-compose.yml logs backend
    exit 1
fi

# Check frontend
frontend_health=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000)

if [ "$frontend_health" -eq 200 ]; then
    echo "✅ Frontend service is healthy"
else
    echo "❌ Frontend service health check failed"
    echo "Checking logs..."
    docker-compose -f docker/docker-compose.yml logs frontend
    exit 1
fi

echo ""
echo "🎉 Deployment completed successfully!"
echo "=================================="
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🏥 Health Check: http://localhost:8000/api/v1/health"
echo ""
echo "To view logs: docker-compose -f docker/docker-compose.yml logs -f"
echo "To stop: docker-compose -f docker/docker-compose.yml down"
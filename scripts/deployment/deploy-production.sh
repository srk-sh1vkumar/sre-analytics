#!/bin/bash
# Deploy SRE Analytics to Production Environment

set -e  # Exit on error

echo "🚀 Starting deployment to PRODUCTION environment..."
echo "⚠️  WARNING: This will deploy to PRODUCTION"
echo ""

# Require confirmation for production
if [ "${PRODUCTION_DEPLOY_CONFIRMED}" != "true" ]; then
    echo "❌ Production deployment requires confirmation"
    echo "   Set PRODUCTION_DEPLOY_CONFIRMED=true to proceed"
    exit 1
fi

# Configuration
NAMESPACE="${NAMESPACE:-sre-analytics}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io}"
DOCKER_IMAGE="${DOCKER_USERNAME}/sre-analytics:${IMAGE_TAG}"

# Deployment method (kubernetes, docker-compose, or helm)
DEPLOY_METHOD="${DEPLOY_METHOD:-kubernetes}"

# Validate image tag is not 'latest' for production
if [ "${IMAGE_TAG}" = "latest" ]; then
    echo "❌ Cannot deploy 'latest' tag to production"
    echo "   Please specify a version tag (e.g., v1.0.0)"
    exit 1
fi

echo "📋 Deployment Configuration:"
echo "  Environment: PRODUCTION"
echo "  Namespace: ${NAMESPACE}"
echo "  Image: ${DOCKER_IMAGE}"
echo "  Method: ${DEPLOY_METHOD}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Deploy using Kubernetes
deploy_kubernetes() {
    echo "☸️  Deploying to Kubernetes..."

    if ! command_exists kubectl; then
        echo "❌ kubectl not found. Please install kubectl."
        exit 1
    fi

    # Create namespace if it doesn't exist
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

    # Apply Kubernetes manifests
    if [ -d "k8s" ]; then
        echo "📦 Applying Kubernetes manifests..."
        kubectl apply -f k8s/ -n "${NAMESPACE}"

        # Update image tag
        kubectl set image deployment/sre-analytics \
            sre-analytics="${DOCKER_IMAGE}" \
            -n "${NAMESPACE}"

        # Wait for rollout
        echo "⏳ Waiting for deployment to complete..."
        kubectl rollout status deployment/sre-analytics -n "${NAMESPACE}" --timeout=10m

        echo "✅ Kubernetes deployment complete"
    else
        echo "⚠️  No k8s/ directory found"
        exit 1
    fi
}

# Deploy using Helm
deploy_helm() {
    echo "⛵ Deploying with Helm..."

    if ! command_exists helm; then
        echo "❌ helm not found. Please install Helm."
        exit 1
    fi

    if [ -d "helm/sre-analytics" ]; then
        helm upgrade --install sre-analytics ./helm/sre-analytics \
            --namespace "${NAMESPACE}" \
            --create-namespace \
            --set image.repository="${DOCKER_REGISTRY}/${DOCKER_USERNAME}/sre-analytics" \
            --set image.tag="${IMAGE_TAG}" \
            --set environment="production" \
            --set replicaCount=3 \
            --wait \
            --timeout=10m

        echo "✅ Helm deployment complete"
    else
        echo "⚠️  No helm/sre-analytics/ directory found"
        exit 1
    fi
}

# Deploy using Docker Compose
deploy_docker_compose() {
    echo "🐳 Deploying with Docker Compose..."

    if ! command_exists docker; then
        echo "❌ docker not found. Please install Docker."
        exit 1
    fi

    if [ -f "docker/docker-compose.production.yml" ]; then
        docker compose -f docker/docker-compose.production.yml pull
        docker compose -f docker/docker-compose.production.yml up -d --remove-orphans

        echo "✅ Docker Compose deployment complete"
    else
        echo "⚠️  No docker/docker-compose.production.yml found"
        exit 1
    fi
}

# Main deployment logic
case "${DEPLOY_METHOD}" in
    kubernetes)
        deploy_kubernetes
        ;;
    helm)
        deploy_helm
        ;;
    docker-compose)
        deploy_docker_compose
        ;;
    *)
        echo "❌ Unknown deployment method: ${DEPLOY_METHOD}"
        echo "   Supported: kubernetes, helm, docker-compose"
        exit 1
        ;;
esac

echo ""
echo "🎉 Production deployment completed successfully!"
echo ""

# Run smoke tests
if [ -f "scripts/deployment/smoke-tests.sh" ]; then
    echo "🧪 Running smoke tests..."
    bash scripts/deployment/smoke-tests.sh production || {
        echo "❌ Smoke tests failed! Consider rollback."
        exit 1
    }
else
    echo "⚠️  No smoke tests found - skipping"
fi

echo ""
echo "📊 Deployment Summary:"
echo "  ✅ Environment: PRODUCTION"
echo "  ✅ Image: ${DOCKER_IMAGE}"
echo "  ✅ Method: ${DEPLOY_METHOD}"
echo ""
echo "🔗 Access production at: https://sre-analytics.example.com"
echo "   (Update URL in .github/workflows/deploy.yml environment configuration)"
echo ""
echo "📢 Next steps:"
echo "  1. Monitor application metrics"
echo "  2. Check error logs"
echo "  3. Verify health endpoints"
echo "  4. Notify stakeholders of deployment"

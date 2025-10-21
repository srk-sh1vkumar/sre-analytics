# SRE Analytics - Deployment Guide

Complete guide for deploying SRE Analytics to staging and production environments.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Health Check Endpoints](#health-check-endpoints)
- [Deployment Methods](#deployment-methods)
- [Staging Deployment](#staging-deployment)
- [Production Deployment](#production-deployment)
- [Smoke Tests](#smoke-tests)
- [Rollback Procedures](#rollback-procedures)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Overview

SRE Analytics supports multiple deployment methods:
- **Kubernetes** (recommended for production)
- **Helm** (Kubernetes with easier configuration)
- **Docker Compose** (development/staging)

The deployment process includes:
1. Docker image build and push
2. Automated deployment to target environment
3. Comprehensive smoke tests
4. Health check validation

---

## Prerequisites

### Required Tools
- Docker (for containerization)
- kubectl (for Kubernetes deployments)
- helm (optional, for Helm deployments)
- bash (for deployment scripts)
- curl (for smoke tests)

### Required Secrets (GitHub Actions)
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub access token

### Optional Secrets
- `SLACK_WEBHOOK_URL` - For deployment notifications
- `KUBERNETES_CONFIG` - Kubeconfig for cluster access

---

## Health Check Endpoints

SRE Analytics provides three health check endpoints:

### 1. `/health` - Basic Health Check
**Purpose**: Simple health status indicator
**Use Case**: Load balancer health checks

```bash
curl http://localhost:5001/health
```

**Response**:
```json
{
  "status": "healthy",
  "service": "sre-analytics",
  "version": "1.0.0",
  "timestamp": "2025-10-20T20:00:00.000Z"
}
```

### 2. `/health/ready` - Readiness Check
**Purpose**: Verify application is ready to serve requests
**Use Case**: Kubernetes readiness probes

```bash
curl http://localhost:5001/health/ready
```

**Response (Ready)**:
```json
{
  "status": "ready",
  "components": {
    "config": "ready",
    "reports_directory": "ready"
  },
  "timestamp": "2025-10-20T20:00:00.000Z"
}
```

**Response (Not Ready)** - HTTP 503:
```json
{
  "status": "not_ready",
  "components": {
    "config": "error: config file not found",
    "reports_directory": "ready"
  },
  "timestamp": "2025-10-20T20:00:00.000Z"
}
```

### 3. `/health/live` - Liveness Check
**Purpose**: Verify application is alive (not deadlocked)
**Use Case**: Kubernetes liveness probes

```bash
curl http://localhost:5001/health/live
```

**Response**:
```json
{
  "status": "alive",
  "timestamp": "2025-10-20T20:00:00.000Z"
}
```

---

## Deployment Methods

### Method 1: Kubernetes (Recommended)

**Requirements**:
- Kubernetes cluster (v1.20+)
- kubectl configured
- Kubernetes manifests in `k8s/` directory

**Deployment**:
```bash
export DOCKER_USERNAME=your-username
export IMAGE_TAG=v1.0.0
export NAMESPACE=sre-analytics
export DEPLOY_METHOD=kubernetes

bash scripts/deployment/deploy-production.sh
```

### Method 2: Helm

**Requirements**:
- Helm 3.0+
- Helm charts in `helm/sre-analytics/` directory

**Deployment**:
```bash
export DOCKER_USERNAME=your-username
export IMAGE_TAG=v1.0.0
export NAMESPACE=sre-analytics
export DEPLOY_METHOD=helm

bash scripts/deployment/deploy-production.sh
```

### Method 3: Docker Compose

**Requirements**:
- Docker Compose v2.0+
- Compose file: `docker/docker-compose.production.yml`

**Deployment**:
```bash
export DOCKER_USERNAME=your-username
export IMAGE_TAG=v1.0.0
export DEPLOY_METHOD=docker-compose

bash scripts/deployment/deploy-production.sh
```

---

## Staging Deployment

### Automatic Deployment (GitHub Actions)

Staging deployment triggers automatically on push to `main` branch:

1. Push to main:
   ```bash
   git push origin main
   ```

2. GitHub Actions workflow:
   - Builds Docker image
   - Pushes to Docker Hub
   - Deploys to staging
   - Runs smoke tests

### Manual Staging Deployment

```bash
# Set environment variables
export DOCKER_USERNAME=your-username
export IMAGE_TAG=main-abc123
export NAMESPACE=sre-analytics-staging
export DEPLOY_METHOD=kubernetes

# Run deployment script
bash scripts/deployment/deploy-staging.sh
```

### Staging Environment Details

- **Namespace**: `sre-analytics-staging`
- **URL**: https://staging.sre-analytics.example.com
- **Image Tag**: `main-{git-sha}`
- **Replicas**: 1

---

## Production Deployment

### Automatic Deployment (GitHub Actions)

Production deployment triggers on version tags:

1. Create and push version tag:
   ```bash
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   ```

2. GitHub Actions workflow:
   - Builds Docker image with version tag
   - Pushes to Docker Hub
   - Deploys to production
   - Runs comprehensive smoke tests
   - Sends deployment notification

### Manual Production Deployment

```bash
# Set environment variables
export DOCKER_USERNAME=your-username
export IMAGE_TAG=v1.0.0
export NAMESPACE=sre-analytics
export DEPLOY_METHOD=kubernetes
export PRODUCTION_DEPLOY_CONFIRMED=true

# Run deployment script
bash scripts/deployment/deploy-production.sh
```

**⚠️ Important**: Production deployment requires:
- `PRODUCTION_DEPLOY_CONFIRMED=true` environment variable
- Specific version tag (not `latest`)

### Production Environment Details

- **Namespace**: `sre-analytics`
- **URL**: https://sre-analytics.example.com
- **Image Tag**: `v{major}.{minor}.{patch}`
- **Replicas**: 3 (for high availability)

---

## Smoke Tests

Smoke tests automatically run after deployment to verify functionality.

### Test Coverage

1. **Health Check** - Verifies `/health` endpoint
2. **Readiness Check** - Verifies `/health/ready` endpoint
3. **Liveness Check** - Verifies `/health/live` endpoint
4. **Homepage** - Verifies main application loads
5. **Response Time** - Measures API performance

### Running Smoke Tests Manually

**Staging**:
```bash
export STAGING_URL=https://staging.sre-analytics.example.com
bash scripts/deployment/smoke-tests.sh staging
```

**Production**:
```bash
export PRODUCTION_URL=https://sre-analytics.example.com
bash scripts/deployment/smoke-tests.sh production
```

### Test Output

```
🧪 Running smoke tests for staging environment...

🔗 Testing endpoint: https://staging.sre-analytics.example.com

1️⃣  Testing /health endpoint...
   ✅ Health check PASSED
   Response: {"status":"healthy","service":"sre-analytics"...}

2️⃣  Testing /health/ready endpoint...
   ✅ Readiness check PASSED

3️⃣  Testing /health/live endpoint...
   ✅ Liveness check PASSED

4️⃣  Testing homepage (/)...
   ✅ Homepage PASSED

5️⃣  Testing API response time...
   ✅ Response time: 245ms (< 1000ms threshold)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ All smoke tests PASSED!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Rollback Procedures

### Kubernetes Rollback

```bash
# View rollout history
kubectl rollout history deployment/sre-analytics -n sre-analytics

# Rollback to previous version
kubectl rollout undo deployment/sre-analytics -n sre-analytics

# Rollback to specific revision
kubectl rollout undo deployment/sre-analytics -n sre-analytics --to-revision=2
```

### Helm Rollback

```bash
# List releases
helm list -n sre-analytics

# View release history
helm history sre-analytics -n sre-analytics

# Rollback to previous release
helm rollback sre-analytics -n sre-analytics

# Rollback to specific revision
helm rollback sre-analytics 2 -n sre-analytics
```

### Docker Compose Rollback

```bash
# Pull previous image version
docker pull your-username/sre-analytics:v1.0.0

# Update docker-compose.yml with previous version
docker compose -f docker/docker-compose.production.yml up -d
```

---

## Monitoring

### Application Metrics

Monitor these key metrics after deployment:

1. **Response Time** - API endpoints should respond < 1s
2. **Error Rate** - Should be < 1%
3. **Memory Usage** - Should be stable
4. **CPU Usage** - Should be < 80%

### Health Checks

Configure your monitoring system to check:
- `/health` - Every 30 seconds
- `/health/ready` - Every 10 seconds
- `/health/live` - Every 30 seconds

### Kubernetes Monitoring

```bash
# Pod status
kubectl get pods -n sre-analytics

# View logs
kubectl logs -f deployment/sre-analytics -n sre-analytics

# Resource usage
kubectl top pods -n sre-analytics

# Events
kubectl get events -n sre-analytics --sort-by='.lastTimestamp'
```

---

## Troubleshooting

### Issue: Deployment Fails

**Check 1: Image availability**
```bash
docker pull ${DOCKER_USERNAME}/sre-analytics:${IMAGE_TAG}
```

**Check 2: Kubernetes secrets**
```bash
kubectl get secrets -n sre-analytics
```

**Check 3: Pod logs**
```bash
kubectl logs -f deployment/sre-analytics -n sre-analytics
```

### Issue: Smoke Tests Fail

**Check 1: Service accessibility**
```bash
curl -v https://sre-analytics.example.com/health
```

**Check 2: DNS resolution**
```bash
nslookup sre-analytics.example.com
```

**Check 3: Ingress configuration**
```bash
kubectl get ingress -n sre-analytics
```

### Issue: Health Checks Fail

**Check 1: Application logs**
```bash
kubectl logs -f deployment/sre-analytics -n sre-analytics | grep health
```

**Check 2: Configuration**
```bash
kubectl describe pod -l app=sre-analytics -n sre-analytics
```

**Check 3: Resource limits**
```bash
kubectl top pods -n sre-analytics
```

### Issue: Slow Response Times

**Check 1: Resource allocation**
```bash
kubectl describe pod -l app=sre-analytics -n sre-analytics | grep -A 5 Limits
```

**Check 2: Database connectivity**
```bash
kubectl exec -it deployment/sre-analytics -n sre-analytics -- curl -v mongodb://...
```

**Check 3: Network latency**
```bash
kubectl exec -it deployment/sre-analytics -n sre-analytics -- ping prometheus-server
```

---

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `DOCKER_USERNAME` | Docker Hub username | Yes | - |
| `IMAGE_TAG` | Docker image tag | Yes | - |
| `NAMESPACE` | Kubernetes namespace | No | `sre-analytics` |
| `DEPLOY_METHOD` | Deployment method | No | `kubernetes` |
| `PRODUCTION_DEPLOY_CONFIRMED` | Production safety check | Yes (prod) | - |
| `STAGING_URL` | Staging base URL | No | `http://localhost:5001` |
| `PRODUCTION_URL` | Production base URL | No | `https://sre-analytics.example.com` |

### Customization

Edit these files to customize deployment:
- `scripts/deployment/deploy-staging.sh` - Staging deployment logic
- `scripts/deployment/deploy-production.sh` - Production deployment logic
- `scripts/deployment/smoke-tests.sh` - Smoke test configuration
- `.github/workflows/deploy.yml` - CI/CD workflow

---

## Best Practices

1. **Version Tags**: Always use semantic versioning (v1.0.0)
2. **Test Staging First**: Always deploy to staging before production
3. **Monitor Deployments**: Watch metrics during and after deployment
4. **Gradual Rollout**: Use canary or blue-green deployments for production
5. **Backup Before Deploy**: Ensure backups are current before production deployment
6. **Communication**: Notify team of production deployments
7. **Rollback Plan**: Always have a rollback plan ready

---

## Related Documentation

- [CI/CD Setup Complete](CI_CD_SETUP_COMPLETE.md)
- [CI/CD Guide](CI_CD_GUIDE.md)
- [API Documentation](../API_DOCUMENTATION.md)

---

**Last Updated**: 2025-10-20
**Version**: 1.0.0

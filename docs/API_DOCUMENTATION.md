# SRE Analytics API Documentation

**Status:** ✅ Complete
**Priority:** 4 (Phase 4: Intelligence & Automation)
**Date:** 2025-10-12

---

## Overview

The SRE Analytics API provides REST

ful access to SLO metrics, anomaly detection, health monitoring, and report generation. Built with FastAPI, it features API key authentication, role-based access control, and rate limiting.

**Base URL:** `http://localhost:8000` (development)

**API Version:** 1.0.0

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-api.txt
```

### 2. Start API Server

```bash
python3 start_api_server.py
```

The server will start on `http://localhost:8000` with:
- **Interactive Docs**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)
- **OpenAPI Spec**: http://localhost:8000/openapi.json

### 3. Get API Key

When the server starts, it creates default API keys:
```
✅ Admin API Key: sre_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   Key ID: xxxxxxxxxxxxxxxx
   Rate Limit: 1000 req/min

✅ Read-Only API Key: sre_yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
   Key ID: yyyyyyyyyyyyyyyy
   Rate Limit: 100 req/min
```

⚠️ **Save these keys!** They cannot be retrieved later.

### 4. Make Your First Request

```bash
# Test health endpoint (no auth required)
curl http://localhost:8000/health

# Test authenticated endpoint
curl -H "X-API-Key: sre_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
     http://localhost:8000/api/v1/status
```

---

## Authentication

### API Key Header

All protected endpoints require an API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
     http://localhost:8000/api/v1/metrics
```

### Roles

- **READ**: Read-only access to metrics, reports, and anomalies
- **WRITE**: Read + Write operations (generate reports, trigger anomaly detection)
- **ADMIN**: Full access including API key management

### Creating API Keys (ADMIN only)

```bash
curl -X POST "http://localhost:8000/api/v1/admin/keys?name=MyApp&role=read&rate_limit=100" \
     -H "X-API-Key: ADMIN_API_KEY"
```

Response:
```json
{
  "api_key": "sre_newkeyxxxxxxxxxxxxxxxx",
  "key_id": "1234567890abcdef",
  "name": "MyApp",
  "role": "read",
  "rate_limit": 100,
  "created_at": "2025-10-12T10:30:00",
  "warning": "Save this API key now! It cannot be retrieved later."
}
```

---

## Rate Limiting

Rate limits are enforced per API key (default: 100 requests/minute).

**Response Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 45
```

**Rate Limit Exceeded (HTTP 429):**
```json
{
  "detail": "Rate limit exceeded. Reset in 45 seconds.",
  "status_code": 429
}
```

---

## Endpoints

### Health & Status

#### GET /health

Health check endpoint (no authentication required).

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-10-12T10:30:00",
  "components": {
    "api": "healthy",
    "authentication": "healthy",
    "rate_limiter": "healthy"
  }
}
```

#### GET /api/v1/status

API status and statistics (requires authentication).

```bash
curl -H "X-API-Key: YOUR_KEY" \
     http://localhost:8000/api/v1/status
```

Response:
```json
{
  "api_version": "1.0.0",
  "authenticated_as": "MyApp",
  "role": "read",
  "rate_limit": {
    "limit": 100,
    "remaining": 95,
    "reset_in": 60
  },
  "timestamp": "2025-10-12T10:30:00"
}
```

---

### Metrics

#### GET /api/v1/metrics

Get current SLO metrics for services.

**Query Parameters:**
- `services` (optional): Comma-separated service names
- `metric_types` (optional): Comma-separated metric types (response_time, error_rate, availability, cpu, memory)

```bash
curl -H "X-API-Key: YOUR_KEY" \
     "http://localhost:8000/api/v1/metrics?services=product-service,order-service"
```

Response:
```json
{
  "services": ["product-service", "order-service"],
  "metrics": [
    {
      "service_name": "product-service",
      "metric_name": "response_time",
      "current_value": 145.5,
      "slo_target": 150.0,
      "sla_target": 200.0,
      "unit": "ms",
      "status": "healthy",
      "error_budget_consumed": 15.3
    }
  ],
  "count": 1,
  "timestamp": "2025-10-12T10:30:00"
}
```

#### GET /api/v1/services/{service_name}/health

Get comprehensive health status for a specific service.

**Path Parameters:**
- `service_name`: Service name

**Query Parameters:**
- `hours_back` (optional): Hours of historical data (default: 1)

```bash
curl -H "X-API-Key: YOUR_KEY" \
     "http://localhost:8000/api/v1/services/product-service/health?hours_back=4"
```

Response:
```json
{
  "service_name": "product-service",
  "health_score": 95.0,
  "status": "healthy",
  "metrics_count": 5,
  "slo_compliance": 98.5,
  "insights": [
    "Response time trending upward",
    "Error rate stable"
  ],
  "recommendations": [
    "Monitor response time closely"
  ],
  "timestamp": "2025-10-12T10:30:00"
}
```

---

### Anomaly Detection

#### GET /api/v1/anomalies

Get recent anomalies across all services.

**Query Parameters:**
- `services` (optional): Comma-separated service names
- `severity` (optional): Filter by severity (info, warning, critical)
- `hours_back` (optional): Hours to look back (default: 4)

```bash
curl -H "X-API-Key: YOUR_KEY" \
     "http://localhost:8000/api/v1/anomalies?severity=critical&hours_back=4"
```

Response:
```json
[
  {
    "service_name": "product-service",
    "metric_name": "response_time",
    "anomalies": [
      {
        "timestamp": "2025-10-12T10:25:00",
        "value": 285.5,
        "expected_value": 145.0,
        "deviation": 140.5,
        "severity": "critical",
        "confidence": 0.95,
        "description": "Modified Z-score 4.56 exceeds threshold 3.75"
      }
    ],
    "baseline_health": "warning",
    "prediction": {
      "will_breach": true,
      "predicted_value": 310.2,
      "slo_target": 150.0,
      "confidence": 0.78,
      "forecast_time": "2025-10-12T10:55:00",
      "trend_slope": 2.5,
      "reason": "Trend slope 2.50 predicts breach"
    },
    "recommendations": [
      "⚠️ 1 critical anomalies detected. Immediate investigation recommended",
      "🔮 Predicted SLO breach in ~25 minutes. Consider scaling up"
    ],
    "detection_method": "modified_z_score",
    "timestamp": "2025-10-12T10:30:00"
  }
]
```

#### POST /api/v1/anomalies/detect

Trigger anomaly detection for specified services (requires WRITE or ADMIN role).

**Request Body:**
```json
{
  "services": ["product-service", "order-service"],
  "detection_method": "modified_z_score",
  "hours_back": 4
}
```

**Detection Methods:**
- `z_score`: Standard Z-score
- `modified_z_score`: MAD-based (default)
- `iqr`: Interquartile range
- `moving_average`: Moving average deviation
- `prophet`: Facebook Prophet (requires fbprophet)

```bash
curl -X POST http://localhost:8000/api/v1/anomalies/detect \
     -H "X-API-Key: YOUR_WRITE_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "services": ["product-service"],
       "detection_method": "modified_z_score",
       "hours_back": 4
     }'
```

---

### Reports

#### GET /api/v1/reports

List generated SRE reports.

**Query Parameters:**
- `limit` (optional): Max reports to return (default: 10)
- `offset` (optional): Skip N reports (default: 0)

```bash
curl -H "X-API-Key: YOUR_KEY" \
     "http://localhost:8000/api/v1/reports?limit=10&offset=0"
```

#### POST /api/v1/reports/generate

Generate a new SRE report (requires WRITE or ADMIN role).

**Request Body:**
```json
{
  "services": ["product-service", "order-service"],
  "start_time": "2025-10-12T00:00:00",
  "end_time": "2025-10-12T23:59:59",
  "include_anomalies": true,
  "include_recommendations": true,
  "format": "json"
}
```

**Formats:** `json`, `html`, `pdf`

```bash
curl -X POST http://localhost:8000/api/v1/reports/generate \
     -H "X-API-Key: YOUR_WRITE_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "services": ["product-service"],
       "start_time": "2025-10-12T00:00:00",
       "end_time": "2025-10-12T23:59:59",
       "include_anomalies": true,
       "include_recommendations": true,
       "format": "json"
     }'
```

Response:
```json
{
  "report_id": "report_20251012_103000",
  "status": "generating",
  "created_at": "2025-10-12T10:30:00",
  "completed_at": null,
  "download_url": null,
  "error": null,
  "timestamp": "2025-10-12T10:30:00"
}
```

#### GET /api/v1/reports/{report_id}

Retrieve a generated report.

```bash
curl -H "X-API-Key: YOUR_KEY" \
     http://localhost:8000/api/v1/reports/report_20251012_103000
```

---

### Admin (API Key Management)

#### POST /api/v1/admin/keys

Create new API key (requires ADMIN role).

**Query Parameters:**
- `name`: Descriptive name
- `role`: Access role (read, write, admin)
- `rate_limit`: Requests per minute

```bash
curl -X POST "http://localhost:8000/api/v1/admin/keys?name=Production&role=write&rate_limit=500" \
     -H "X-API-Key: ADMIN_KEY"
```

#### GET /api/v1/admin/keys

List all API keys (requires ADMIN role).

```bash
curl -H "X-API-Key: ADMIN_KEY" \
     http://localhost:8000/api/v1/admin/keys
```

Response:
```json
[
  {
    "key_id": "1234567890abcdef",
    "name": "Production",
    "role": "write",
    "created_at": "2025-10-12T10:00:00",
    "last_used": "2025-10-12T10:30:00",
    "rate_limit": 500,
    "enabled": true,
    "metadata": {}
  }
]
```

#### DELETE /api/v1/admin/keys/{key_id}

Revoke an API key (requires ADMIN role).

```bash
curl -X DELETE http://localhost:8000/api/v1/admin/keys/1234567890abcdef \
     -H "X-API-Key: ADMIN_KEY"
```

---

## Python Client Example

```python
import requests

class SREAnalyticsClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key}

    def get_metrics(self, services: list = None):
        """Get current metrics"""
        params = {"services": ",".join(services)} if services else {}
        response = requests.get(
            f"{self.base_url}/api/v1/metrics",
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()

    def get_service_health(self, service_name: str, hours_back: int = 1):
        """Get service health"""
        response = requests.get(
            f"{self.base_url}/api/v1/services/{service_name}/health",
            headers=self.headers,
            params={"hours_back": hours_back}
        )
        response.raise_for_status()
        return response.json()

    def get_anomalies(self, severity: str = None, hours_back: int = 4):
        """Get recent anomalies"""
        params = {"hours_back": hours_back}
        if severity:
            params["severity"] = severity

        response = requests.get(
            f"{self.base_url}/api/v1/anomalies",
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()

# Usage
client = SREAnalyticsClient(
    base_url="http://localhost:8000",
    api_key="sre_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)

# Get metrics
metrics = client.get_metrics(services=["product-service"])
print(f"Found {metrics['count']} metrics")

# Get service health
health = client.get_service_health("product-service", hours_back=4)
print(f"Health score: {health['health_score']}/100")

# Get anomalies
anomalies = client.get_anomalies(severity="critical")
print(f"Found {len(anomalies)} critical anomalies")
```

---

## Error Handling

All errors follow a consistent format:

```json
{
  "error": "Error message",
  "status_code": 404,
  "timestamp": "2025-10-12T10:30:00",
  "details": {}
}
```

**Common Status Codes:**
- `200 OK`: Success
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid API key
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

---

## Configuration

### Environment Variables

- `API_HOST`: Host to bind to (default: 0.0.0.0)
- `API_PORT`: Port to listen on (default: 8000)
- `API_RELOAD`: Enable auto-reload (default: true)

```bash
export API_HOST=0.0.0.0
export API_PORT=8080
export API_RELOAD=false
python3 start_api_server.py
```

---

## Production Deployment

### 1. Install Production Dependencies

```bash
pip install -r requirements-api.txt
pip install gunicorn  # Production WSGI server
```

### 2. Run with Gunicorn

```bash
gunicorn src.api.app:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --access-logfile - \
    --error-logfile -
```

### 3. Behind Reverse Proxy (Nginx)

```nginx
upstream sre_api {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name api.sre-analytics.example.com;

    location / {
        proxy_pass http://sre_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 4. Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Best Practices

### 1. Secure API Keys

- Store API keys in environment variables or secrets management
- Never commit API keys to version control
- Rotate keys regularly
- Use separate keys for different environments

### 2. Rate Limiting

- Start with conservative rate limits
- Monitor rate limit headers
- Implement exponential backoff for retries
- Use WRITE keys only when needed

### 3. Error Handling

- Always check HTTP status codes
- Log API errors for debugging
- Implement retry logic with backoff
- Handle rate limiting gracefully

### 4. Monitoring

- Monitor API response times
- Track rate limit usage
- Alert on authentication failures
- Monitor error rates per endpoint

---

## Limitations

- **In-Memory Storage**: API keys stored in memory (use database in production)
- **No Persistence**: Keys lost on restart (implement database backend)
- **Single Instance**: No distributed rate limiting (use Redis in production)
- **No Webhooks**: Async report generation needs polling
- **Basic Auth**: Only API key auth (add OAuth2/JWT for production)

---

## Future Enhancements

- [ ] Database-backed API key storage (PostgreSQL/MongoDB)
- [ ] Redis-based distributed rate limiting
- [ ] OAuth2/JWT authentication
- [ ] Webhook notifications for async operations
- [ ] GraphQL API support
- [ ] WebSocket streaming for real-time metrics
- [ ] API usage analytics dashboard
- [ ] SDK generation for multiple languages

---

## References

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **OpenAPI Spec**: http://localhost:8000/openapi.json
- **Interactive Docs**: http://localhost:8000/docs

---

**Version:** 1.0.0
**Last Updated:** 2025-10-12
**Maintainers:** SRE Analytics Team

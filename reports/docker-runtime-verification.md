# Docker Runtime Verification — MP077

**Date:** 26 November 2026  
**Contributor:** Animesh Yadav  
**GitHub Username:** Animesh160604  
**Branch:** docker-kubernetes-cd

## Objective

To verify the Dockerized FastAPI application and ensure that the trained machine learning model performs inference correctly inside a Docker container.

## Environment

- Operating system: Windows with WSL2
- Docker Engine: 29.8.2
- Python base image: python:3.12-slim
- Docker image: mp077-api:1.0
- Container name: mp077-container
- Application port: 8000

## Docker Image Build

The Docker image was built successfully using the existing project Dockerfile.

- Image ID: ac81517bbef0
- Disk usage: 1.46 GB
- Content size: 328 MB
- Build status: Successful

## API Runtime Test Results

| Test | Expected Result | Actual Result | Status |
|---|---|---|---|
| Docker image build | Successful build | Successful | PASS |
| Container startup | Running | Running | PASS |
| FastAPI startup | Successful | Successful | PASS |
| GET /health | HTTP 200 | HTTP 200 | PASS |
| Valid POST /predict | HTTP 200 | HTTP 200 | PASS |
| Nearby clinics = 26 | HTTP 422 | HTTP 422 | PASS |
| Discount percentage = -5 | HTTP 422 | HTTP 422 | PASS |
| Structured JSON logging | Prediction event logged | Verified | PASS |

## Prediction Result

The prediction endpoint returned 233.02902850216827 medicine units for the sample valid input.

## Input Validation

The API successfully rejected a clinic count greater than 25 and a negative discount percentage.

## Structured Logging

The Docker logs confirmed that successful predictions generated structured JSON logs containing a timestamp, log level, event name and prediction value.

## Conclusion

The existing Dockerized FastAPI application was successfully built, executed and tested locally. API health, model inference, input validation and structured logging were verified successfully.


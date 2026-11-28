# Kubernetes Runtime Verification — MP077

**Date:** 27–28 November 2026

**Contributor:** Animesh Kumar Yadav

**GitHub Username:** Animesh160604

**Branches:** docker-kubernetes-cd, fix-docker-image-tag

## Objective

To verify the deployment of the Dockerized MP077 FastAPI machine learning application in Kubernetes, including replica availability, API functionality, health monitoring, automatic recovery and resource utilization.

## Environment

- Operating System: Windows with WSL2
- Platform: Docker Desktop
- Cluster Type: Kubeadm
- Kubernetes Version: v1.36.1
- Kubernetes Context: docker-desktop
- Number of Nodes: 1
- Namespace: default
- Docker Image: mp077-api:2.0
- Registered Model: mp077-model
- Model Alias: champion
- Champion Model Version: 2 (as recorded in model metadata)

## Deployment Configuration

| Parameter | Value |
|---|---|
| Deployment Name | mp077-api |
| Number of Replicas | 4 |
| Container Port | 8000 |
| Image Pull Policy | IfNotPresent |
| Deployment Strategy | RollingUpdate |
| Service Name | mp077-svc |
| Service Type | ClusterIP |
| Service Port | 80 |
| Target Port | 8000 |

The existing Kubernetes deployment and service manifests were inspected and successfully applied without unnecessary modifications.

## Runtime Verification Results

| Test | Expected Result | Actual Result | Status |
|---|---|---|---|
| Kubernetes Cluster | Active | Active | PASS |
| Kubernetes Node | Ready | Ready | PASS |
| Deployment Rollout | Successful | Successful | PASS |
| Application Replicas | 4/4 Ready | 4/4 Ready | PASS |
| Pod Status | Running | All Running | PASS |
| Container Restarts | 0 | 0 | PASS |
| Service Creation | Successful | Successful | PASS |
| Health Endpoint | HTTP 200 | HTTP 200 | PASS |
| Prediction Endpoint | HTTP 200 | HTTP 200 | PASS |
| Service Endpoints | 4 | 4 | PASS |
| Automatic Pod Replacement | Successful | Successful | PASS |
| Automated API Smoke Tests | 4/4 Passed | 4/4 Passed | PASS |

## API Verification

The Kubernetes Service was accessed through local port forwarding using port 8080.

The GET /health endpoint returned HTTP 200 with the response:

`{"status":"ok"}`

The POST /predict endpoint successfully generated a prediction of 973.0170133892688 medicine units using the default Swagger request values.

The automated smoke testing script subsequently returned a prediction of 233.02902850216827 medicine units using the same sample input previously tested in Docker.

The matching prediction results verified consistent model inference across Docker and Kubernetes for that input.

## Health Probe Configuration

| Parameter | Readiness Probe | Liveness Probe |
|---|---|---|
| Endpoint | /health | /health |
| Initial Delay | 5 seconds | 15 seconds |
| Check Interval | 10 seconds | 20 seconds |
| Timeout | 2 seconds | 2 seconds |
| Failure Threshold | 3 | 3 |

Application logs confirmed repeated successful HTTP 200 health responses across all four application pods.

## Kubernetes Self-Healing Verification

The following application pod was intentionally deleted:

`mp077-api-7c897cb685-gjtj7`

Kubernetes automatically created the following replacement pod:

`mp077-api-7c897cb685-jxrh2`

After automatic recovery:

- Desired replicas: 4
- Available replicas: 4
- Unavailable replicas: 0
- All application pods: Running
- Container restarts: 0

This demonstrated Kubernetes automatic replica reconciliation and recovery following pod deletion.

## Resource Utilization

| Pod Suffix | Memory Usage | CPU Usage |
|---|---|---|
| jxrh2 | 129.6 MiB | 0.18% |
| j5bct | 140.3 MiB | 0.18% |
| z6sds | 141.3 MiB | 0.17% |
| ss2ns | 142.4 MiB | 0.17% |
| Total | 553.6 MiB | 0.70% |

Resource utilization was measured using the docker stats command. These measurements represent the application containers only and exclude Kubernetes system overhead.

## Automated Smoke Testing

A reusable Python script was implemented at:

`.github/scripts/api_smoke_test.py`

The script was executed against the Kubernetes deployment.

| Automated Test | Result |
|---|---|
| Health Endpoint | PASS |
| Valid Prediction | PASS |
| Invalid Clinic Count | PASS |
| Invalid Discount Percentage | PASS |

All four automated API smoke tests passed successfully.

## Commands Executed

- docker build -t mp077-api:2.0 .
- kubectl get nodes
- kubectl apply -f k8s/
- kubectl rollout status deployment/mp077-api --timeout=300s
- kubectl get deployments
- kubectl get pods -o wide
- kubectl get services
- kubectl port-forward service/mp077-svc 8080:80
- kubectl delete pod mp077-api-7c897cb685-gjtj7
- kubectl get pods -w
- kubectl describe deployment mp077-api
- kubectl logs -l app=mp077-api --prefix=true --tail=20
- kubectl get endpointslices -l kubernetes.io/service-name=mp077-svc
- docker stats --no-stream
- python .github/scripts/api_smoke_test.py http://127.0.0.1:8080

## Conclusion

The MP077 FastAPI machine learning application was successfully deployed and verified in a local Kubernetes cluster.

All four application replicas were running and available. Health monitoring, prediction functionality, input validation, service connectivity and automatic pod replacement were successfully verified.

The automated API smoke testing script passed all four tests against the Kubernetes deployment, demonstrating successful runtime integration of the Dockerized machine learning application with Kubernetes.


## Final Docker Image Tag Compliance Verification — 28 November 2026

The Docker image tag was changed from `mp077-api:2.0` to `mp077-api:1.0` to comply with the official MP077 problem specification.

### Configuration Changes

- Updated `k8s/deployment.yaml` to use `mp077-api:1.0`.
- Updated Docker build, container execution and Kind image loading commands in `.github/workflows/ci.yml`.
- Updated the README deployment instructions.
- Retained MLflow Champion V2 without modifying the trained model or holdout predictions.

### Runtime Verification Results

| Verification | Result |
|---|---|
| Docker image build | PASS |
| Required image tag | mp077-api:1.0 |
| Champion model version | V2 |
| Docker API smoke tests | 4/4 passed |
| Kubernetes deployment rollout | Successful |
| Kubernetes replicas | 4/4 ready |
| Pod restarts at verification | 0 |
| Kubernetes API smoke tests | 4/4 passed |
| Valid prediction from both environments | 233.02902850216827 |
| Invalid clinic count rejection | PASS |
| Invalid discount rejection | PASS |

The corrected Docker image and Kubernetes deployment were successfully verified. The image tag correction did not change the tested model prediction or API validation behavior.

The previous verification results using image tag `2.0` are retained as historical testing evidence.




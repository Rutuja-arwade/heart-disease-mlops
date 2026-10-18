# Heart Disease Risk — End-to-End MLOps Pipeline

[![ci-cd](https://github.com/Rutuja-arwade/heart-disease-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Rutuja-arwade/heart-disease-mlops/actions/workflows/ci.yml)

A classifier that predicts the presence of heart disease from 13 clinical attributes (UCI Heart Disease,
Cleveland subset). It is served as a monitored REST API, packaged in Docker and deployed to Kubernetes.

**Course:** MLOps (AIMLCZG523), Assignment 1 · **Author:** Rutuja Arwade

| Stage | Implementation |
|---|---|
| Data | `src/data.py`: checksum-verified download with mirror fallback, validation and cleaning |
| EDA | `src/eda.py` and `notebooks/01_eda.ipynb`: 8 figures in `reports/figures/` |
| Features | `src/features.py`: custom `ClinicalFeatureBuilder` + `ColumnTransformer` (impute, scale, one-hot) in one sklearn `Pipeline` |
| Models | Logistic Regression, Random Forest and Gradient Boosting (plus a majority baseline), tuned with stratified 5-fold CV |
| Tracking | MLflow: one parent run per model, one nested run per tuning trial, plus plots, artifacts and a model registry entry (`champion` alias) |
| Packaging | `models/model.joblib` (full pipeline), `models/mlflow_model/` (MLflow flavour), `models/model_metadata.json` |
| Serving | FastAPI: `/predict`, `/predict/batch`, `/health`, `/ready`, `/model/info`, `/metrics` |
| CI/CD | GitHub Actions: lint → test → train (+ quality gate) → Docker build/test → kind + Helm deploy → GHCR publish |
| Deploy | Minikube with a Helm chart (`deploy/helm`) or raw manifests (`deploy/k8s`), exposed via LoadBalancer + NGINX Ingress, with HPA and PDB |
| Monitoring | JSON request logs, Prometheus metrics, Grafana dashboard and alert rules (docker-compose and Kubernetes) |

**Results:** the champion is L2-regularised logistic regression. Mean CV ROC-AUC is 0.907 ± 0.019. On the held-out test set: ROC-AUC 0.961, accuracy 0.869, recall 0.929.

---

## 1. Quick start (clean machine)

Prerequisites: Python 3.11, Docker. For the deployment steps you also need minikube, kubectl and helm.

```bash
git clone https://github.com/Rutuja-arwade/heart-disease-mlops.git
cd heart-disease-mlops
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt        # or: conda env create -f environment.yml

python -m src.data      # 1. download + clean -> data/processed/heart_disease_clean.csv
python -m src.eda       # 2. EDA figures -> reports/figures/
python -m src.train     # 3. tune + track (MLflow) + export champion -> models/
pytest                  # 4. 45 unit tests
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # browse experiments
```

`make help`-style shortcuts are in the [Makefile](Makefile): `make data eda train test lint`.

## 2. Run the API

```bash
# local (dev)
uvicorn src.api.app:app --port 8000
# container
docker build -t heart-disease-api:1.0.0 .
docker run -d --name heart-api -p 8000:8000 heart-disease-api:1.0.0
./scripts/smoke_test.sh http://localhost:8000
```

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```

```json
{"request_id":"97606faf19c640b8","prediction":1,"label":"heart_disease","confidence":0.9978,
 "probability_disease":0.9978,"risk_level":"high","threshold":0.5,"model_version":"1.0.0+logistic_regression"}
```

Interactive docs: http://localhost:8000/docs. `ca` and `thal` are optional. When they are missing, the pipeline imputes them.
Invalid payloads return HTTP 422 with field-level errors.

A prebuilt image is published by CI: `docker pull ghcr.io/rutuja-arwade/heart-disease-api:latest`.

## 3. Monitoring stack (docker-compose)

```bash
docker compose up -d --build                         # API :8000, Prometheus :9090, Grafana :3000
python scripts/load_test.py --requests 500           # generate traffic
open http://localhost:3000                           # dashboard "Heart Disease API" (admin / heart-admin)
```

## 4. Kubernetes deployment (Minikube)

```bash
minikube start --driver=docker --cpus=2 --memory=3900
minikube addons enable ingress && minikube addons enable metrics-server
minikube image load heart-disease-api:1.0.0          # or use the GHCR image

helm upgrade --install heart deploy/helm/heart-disease-api -n heart-disease --create-namespace --wait
# alternative without Helm:  kubectl apply -k deploy/k8s
kubectl apply -k monitoring                          # Prometheus + Grafana in namespace "monitoring"

minikube tunnel                                      # (separate terminal) assigns the LoadBalancer IP
kubectl get svc -n heart-disease                     # EXTERNAL-IP 127.0.0.1, port 8080
curl -X POST http://127.0.0.1:8080/predict -H "Content-Type: application/json" -d @sample_request.json
helm test heart -n heart-disease                     # in-cluster /predict check
```

Ingress: `heart-api.local` is routed by NGINX. To test without editing `/etc/hosts`:
`kubectl -n ingress-nginx port-forward svc/ingress-nginx-controller 8081:80` and then
`curl -H "Host: heart-api.local" localhost:8081/health`.

## 5. Repository layout

```
├── src/                    data.py, eda.py, features.py, train.py, evaluate.py, predict.py, config.py
│   └── api/                app.py (FastAPI), schemas.py (pydantic), monitoring.py (Prometheus + JSON logs)
├── data/raw|processed/     raw UCI file, cleaned CSV, cleaning report
├── notebooks/              01_eda, 02_model_training, 03_inference (executed)
├── test/                   45 pytest tests (data, features, model, API)
├── models/                 champion pipeline (joblib + MLflow format) and metadata
├── reports/                figures/ and metrics/ generated by the pipeline
├── Dockerfile, docker-compose.yml, .dockerignore
├── deploy/k8s/             Namespace, ConfigMap, Deployment, Service(LB), Ingress, HPA, PDB
├── deploy/helm/heart-disease-api/   Helm chart (+ values-ci.yaml, helm test hook)
├── monitoring/             Prometheus config + alert rules, Grafana provisioning + dashboard, k8s manifests
├── .github/workflows/ci.yml
├── scripts/                download_data.sh, smoke_test.sh, load_test.py
├── screenshots/            evidence used in the report
└── docs/report/            final report (REPORT.md draft, .docx, .pdf) + architecture diagram
```

## 6. CI/CD

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and pull request to `main`. The jobs are:

1. Lint (ruff, helm lint, rendered manifests).
2. Unit tests (pytest with JUnit and coverage).
3. Training with MLflow, followed by a quality gate that fails when CV or test ROC-AUC is below 0.85 or recall is below 0.75.
4. Docker build and container smoke test.
5. kind cluster + Helm install + `helm test` + smoke test.
6. Push to GHCR.

Each job uploads its logs and artifacts. Any failing step stops the pipeline. For example, [PR #1](https://github.com/Rutuja-arwade/heart-disease-mlops/pull/1)
introduced a deliberate bug: the unit tests failed, and training, image build and deployment were skipped.

## 7. Report, screenshots and video

| Deliverable | Location |
|---|---|
| Final report | [docs/report/report.pdf](docs/report/report.pdf) · [report.docx](docs/report/report.docx) · Markdown draft [REPORT.md](docs/report/REPORT.md) |
| Architecture diagram | [docs/report/architecture.png](docs/report/architecture.png) |
| Screenshots (per task) | [screenshots/](screenshots/README.md) |
| Pipeline walkthrough video | [docs/video/pipeline_walkthrough.mp4](docs/video/pipeline_walkthrough.mp4) |

Rebuild the report with `python docs/report/build_report.py` (needs pandoc and Google Chrome).

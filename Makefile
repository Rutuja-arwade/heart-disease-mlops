# Convenience targets. Every target is a thin wrapper around a documented command.
PY ?= python
IMAGE ?= heart-disease-api:1.0.0
RELEASE ?= heart
NAMESPACE ?= heart-disease

.PHONY: setup data eda train train-quick test lint format serve mlflow-ui notebooks \
        docker-build docker-run smoke compose-up compose-down load \
        k8s-image helm-deploy k8s-apply monitoring-k8s k8s-status clean

setup:            ## create venv and install pinned dependencies
	python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt

data:             ## download + clean the UCI dataset
	$(PY) -m src.data

eda:              ## generate EDA figures into reports/figures
	$(PY) -m src.eda

train:            ## tune LR / RF / GB with MLflow tracking, export champion to models/
	$(PY) -m src.train

train-quick:      ## single-candidate smoke run (does not overwrite models/)
	$(PY) -m src.train --quick --no-register

test:             ## unit tests with coverage
	pytest -v --cov=src --cov-report=term-missing

lint:             ## ruff lint + format check
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

serve:            ## run the API locally with autoreload
	uvicorn src.api.app:app --reload --port 8000

mlflow-ui:        ## browse experiments at http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

notebooks:        ## execute notebooks in place
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run -d --name heart-api -p 8000:8000 $(IMAGE)

smoke:            ## smoke test a running API (default http://localhost:8000)
	./scripts/smoke_test.sh $(URL)

compose-up:       ## API + Prometheus + Grafana
	docker compose up -d --build

compose-down:
	docker compose down -v

load:             ## generate traffic for the dashboards
	$(PY) scripts/load_test.py --requests 500 --concurrency 8

k8s-image:        ## make the local image available inside minikube
	minikube image load $(IMAGE)

helm-deploy:      ## deploy/upgrade the API with Helm
	helm upgrade --install $(RELEASE) deploy/helm/heart-disease-api -n $(NAMESPACE) --create-namespace --wait

k8s-apply:        ## alternative: plain manifests via kustomize
	kubectl apply -k deploy/k8s

monitoring-k8s:   ## Prometheus + Grafana in the cluster
	kubectl apply -k monitoring

k8s-status:
	kubectl get all,ingress,hpa -n $(NAMESPACE)

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov

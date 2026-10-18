# Screenshots

Evidence for the report, grouped by assignment task. Browser pages (MLflow, Swagger, Grafana, Prometheus,
GitHub) were captured with headless Chrome. Terminal images show the real output of the command printed in each image,
run on the project. CI log images come from `gh run view --log` for the linked workflow runs.

| # | File | Shows |
|---|---|---|
| 00 | `00_architecture_diagram.png` | End-to-end architecture |
| 01 | `01_github_repository.png` | Public repository |
| **Task 1 · Data & EDA** | | |
| 03 | `03_data_download_and_cleaning.png` | `python -m src.data --force`: download, checksum, cleaning report |
| 04 | `04_notebook_eda_on_github.png` | Executed EDA notebook rendered on GitHub |
| **Task 2-3 · Modelling & MLflow** | | |
| 05 | `05_mlflow_experiment_runs.png` | Experiment `heart-disease-risk`, one parent run per model family |
| 05b | `05b_mlflow_nested_trial_runs.png` | Nested tuning-trial runs under the logistic-regression parent |
| 05c | `05c_mlflow_chart_view.png` | ROC-AUC charts across model families |
| 06 | `06_mlflow_run_overview.png` | Champion run: tags, params, metrics, registered model |
| 07 | `07_training_mlflow_run.png` | `python -m src.train` console output |
| 08 | `08_mlflow_model_metrics.png` | Metric charts of the champion run |
| 09 | `09_mlflow_artifacts_plots.png` | Logged artifacts (plots, cv_results, model) |
| 10 | `10_mlflow_compare_runs.png` | Run comparison (parallel coordinates, params, metrics) |
| 11 | `11_mlflow_model_registry.png`, `11b_mlflow_model_version_champion.png` | Model registry, version 1 with alias `@champion` |
| **Task 5 · Tests & CI/CD** | | |
| 12 | `12_unit_tests_pytest.png` | 45 tests passing, 89 % coverage |
| 13 | `13_lint_ruff_helm.png` | ruff + helm lint |
| 28 | `28_github_actions_pipeline.png` | Successful 6-job workflow run |
| 28b | `28b_github_actions_runs.png` | Workflow run history |
| 29 | `29_ci_test_job_log.png` | CI log: pytest job |
| 29a | `29a_github_actions_test_job_steps.png` | Steps of the CI test job |
| 29b | `29b_ci_k8s_deploy_job_log.png` | CI log: kind cluster, Helm install, helm test, smoke test |
| 29c | `29c_ci_training_job_log.png` | CI log: training and quality gate |
| 29d | `29d_ci_docker_job_log.png` | CI log: container smoke test |
| 29e | `29e_github_actions_deploy_job_steps.png` | Steps of the CI deploy job |
| 30 | `30_github_actions_artifacts.png` | Artifacts uploaded by a workflow run |
| 32 | `32_ci_failure_demo.png`, `32b_ci_failure_demo_run.png` | PR #1 with a deliberate bug: unit tests fail and downstream jobs are skipped |
| **Task 6 · Docker** | | |
| 14 | `14_docker_build.png` | Image build |
| 15 | `15_docker_run_smoke_test.png` | Container run + smoke test with sample input |
| 16 | `16_docker_predict_curl.png` | `/predict` response and the matching JSON log line |
| 17 | `17_swagger_ui.png`, `17b_swagger_predict_response.png` | Interactive API docs and a live prediction |
| 31 | `31_ghcr_container_image.png` | Image published to GitHub Container Registry |
| **Task 7 · Kubernetes** | | |
| 18 | `18_k8s_resources.png` | Nodes, deployment, pods, LoadBalancer service, ingress, HPA, PDB, monitoring pods |
| 19 | `19_k8s_endpoint_verification.png` | `/health` and `/predict` through the LoadBalancer and the Ingress |
| 20 | `20_helm_release_and_test.png` | Helm release, `helm test`, rollout history |
| 23 | `23_load_test.png` | Load test against the cluster |
| **Task 8 · Monitoring & logging** | | |
| 21 | `21_k8s_structured_logs.png` | JSON request / prediction / validation logs |
| 22 | `22_prometheus_metrics_endpoint.png` | Custom metrics exposed at `/metrics` |
| 24 | `24_grafana_dashboard_k8s.png` | Grafana dashboard (Kubernetes) |
| 25 | `25_prometheus_targets_k8s.png`, `25b_prometheus_alert_rules.png`, `25c_prometheus_query_graph.png` | Scrape targets, alert rules, a PromQL query |
| 26 | `26_docker_compose_stack.png` | docker-compose stack (API + Prometheus + Grafana) |
| 27 | `27_grafana_dashboard_compose.png` | Grafana dashboard (docker-compose) |

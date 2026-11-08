# Hotel Booking Cancellation Prediction — End-to-End MLOps

**Real dataset:** https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset

**Problem:** Predict `booking_status` (`Canceled` vs `Not_Canceled`) from the 17 booking attributes available at reservation time. Source CSV has 36,275 historical reservations and 19 columns; `Booking_ID` and the target are not fed into the classifier. Original observations refer to 2017–2018 and do not represent current hotel patterns.

**Labs mapped:** Lab 1 (training + MLflow), Lab 2 (DVC/S3 + FastAPI + Docker/CI/CD), Lab 3 (Kubernetes), Lab 4 (Prometheus + input drift monitoring). Dataset is **not bundled**; retrieve the genuine public Kaggle file using the provided downloader.

## Complete Windows PowerShell runbook

> Requirements: Windows, Python 3.11, Git, Docker Desktop, optional AWS CLI + AWS S3 account. Do not upload AWS secrets or private tokens to GitHub or AI chat. AWS S3 may incur charges. Run commands from the project root after unzipping. Do not put the ZIP itself inside a Git repo.

### 1. Prepare Python environment

```powershell
cd "$HOME\Desktop\hotel-cancellation-mlops"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "dvc[s3]>=3,<4"
```

If activation is blocked, run `Set-ExecutionPolicy -Scope Process Bypass` and try again. If `py -3.11` is missing, install Python 3.11.

### 2. Fetch ORIGINAL Kaggle dataset (internet needed)

```powershell
python scripts/download_data.py
python -c "import pandas as pd; d=pd.read_csv('data/hotel_reservations.csv'); print(d.shape); print(d.booking_status.value_counts())"
```

Expected source shape: `(36275, 19)`. The download script locates the named public Kaggle CSV via `kagglehub`, then stores it in `data/hotel_reservations.csv`. If Kaggle requires sign-in, log into Kaggle on your computer and follow KaggleHub authentication instructions, or download the CSV manually from the above dataset link and save it with that path/name.

### 3. Train, compare models, record in MLflow

```powershell
python train.py
python scripts/quality_gate.py
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open **http://127.0.0.1:5000** to compare logistic regression, random forest, and histogram gradient boosting. Training saves `artifacts/champion.pkl` and `artifacts/metrics.json` (accuracy, precision, recall, F1, ROC-AUC). The champion has the highest held-out F1 (ROC-AUC tie-breaker). Gate requires F1 >= 0.65 and ROC-AUC >= 0.75; these are **proposed demo criteria**, not verified achieved scores. If results fail, investigate instead of silently lowering thresholds.

### 4. Run FastAPI and its tests

Stop MLflow with Ctrl+C, or open a second PowerShell terminal and reactivate `.venv`.

```powershell
python -m pytest -q
python -m uvicorn app:app --reload --port 8000
```

Visit **http://127.0.0.1:8000/docs**, open `POST /predict`, choose "Try it out", enter the sample below, click Execute. The output is a cancellation probability and a label.

```json
{
  "no_of_adults": 2,
  "no_of_children": 0,
  "no_of_weekend_nights": 1,
  "no_of_week_nights": 2,
  "type_of_meal_plan": "Meal Plan 1",
  "required_car_parking_space": 0,
  "room_type_reserved": "Room_Type 1",
  "lead_time": 90,
  "arrival_year": 2018,
  "arrival_month": 10,
  "arrival_date": 12,
  "market_segment_type": "Online",
  "repeated_guest": 0,
  "no_of_previous_cancellations": 0,
  "no_of_previous_bookings_not_canceled": 0,
  "avg_price_per_room": 110.0,
  "no_of_special_requests": 1
}
```

Other endpoints: `GET /health`, `GET /metrics`. Requests are logged to `predictions.csv` (timestamps, lead time, predicted probability; not personally identifying guest data).

### 5. Git + DVC + your S3 bucket

Create **your own private S3 bucket** using AWS Console. Keep public access blocked. Set up **least-privilege IAM** with access to only that bucket. Configure the **AWS CLI** under a profile on your computer (or use temporary SSO credentials where supported):

```powershell
aws configure --profile dvc
aws sts get-caller-identity --profile dvc
```

Replace `YOUR_UNIQUE_BUCKET_NAME` and region with your real details:

```powershell
git init -b main
python -m dvc init
python -m dvc remote add -d storage s3://YOUR_UNIQUE_BUCKET_NAME/dvc
python -m dvc remote modify storage region ap-south-1
python -m dvc remote modify --local storage profile dvc
python -m dvc add data/hotel_reservations.csv
python -m dvc push
```

The real CSV is in S3, the small `data/hotel_reservations.csv.dvc` pointer goes to Git. Commit `.dvc/config` but **never** `.dvc/config.local` or AWS secrets. Verify with `python -m dvc status -c`.

To demonstrate versioning, make an **approved real-data** revision of the CSV, then run:

```powershell
python -m dvc add data/hotel_reservations.csv
python -m dvc push
git add .
git commit -m "Update versioned hotel data"
```

**Never invent hotel cancellations as ground-truth labels** to make retraining appear real. Restore a tracked version via Git + `python -m dvc pull`.

### 6. Publish code to GitHub

Create an empty **private** GitHub repository called `hotel-cancellation-mlops`, no starter README. Replace `YOUR_GITHUB_USERNAME`:

```powershell
git add .
git commit -m "Initial end-to-end hotel cancellation MLOps project"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/hotel-cancellation-mlops.git
git push -u origin main
```

GitHub repo -> **Settings -> Secrets and variables -> Actions**. Set secrets `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, and repository variable `AWS_REGION` (e.g. `ap-south-1`). The IAM principal needs `s3:ListBucket`, `s3:GetObject` on your specific private S3 bucket. For S3 uploads locally, it also needs `s3:PutObject`. GitHub OIDC with short-lived AWS permissions is preferred for a real deployment; static secrets are included for lab simplicity.

GitHub -> **Actions** -> Hotel Cancellation ML Pipeline -> **Run workflow**. On push, GitHub will automatically: install packages -> `dvc pull` -> train 3 models + MLflow -> enforce quality gate -> pytest -> upload metrics artifact -> build/push Docker image. **A GitHub pipeline cannot see your laptop's MLflow runs**: it uses a fresh temporary local MLflow store. For a persistent shared MLflow UI, configure a remote tracking server.

### 7. Docker Hub

Create a Docker Hub account and repository `hotel-cancellation-api`. Create a **scoped read/write token** for GitHub Actions. Never paste it into code. After the pipeline succeeds, open Docker Hub and expect tags such as `build-1` and `latest`.

Local test without waiting for CI:

```powershell
docker build -t hotel-cancellation-api:local .
docker run --rm -p 8000:8000 hotel-cancellation-api:local
```

Use http://127.0.0.1:8000/docs (stop the local Uvicorn process first to free port 8000).

### 8. Monitoring + data drift

Local API running: open **http://127.0.0.1:8000/metrics**; each `/predict` increments Prometheus metrics.

For full Prometheus Docker demo, stop the local API first, create a blank log file if missing, then:

```powershell
if (!(Test-Path predictions.csv)) { New-Item predictions.csv -ItemType File | Out-Null }
docker compose up --build
```

Open **http://127.0.0.1:9090**. In Prometheus, query `hotel_predictions_total` or `rate(hotel_predictions_total[1m])`. API docs remain at http://127.0.0.1:8000/docs. For this demo, the bind mount makes the container write predictions to the same host-side CSV.

After at least 30 predictions, in another terminal run:

```powershell
python scripts/check_drift.py
```

It compares recent `lead_time` values against the original data using a **Kolmogorov-Smirnov test**. A p-value below 0.05 triggers exit code 1, which can drive an alert. **This only checks input drift for one feature; it does not prove the model's accuracy has dropped.** Fresh, legitimately labeled data is required to retrain and assess model quality. After review, commit the new DVC pointer + `dvc push` + `git push`, which runs the training CI job again.

### 9. Optional: Kubernetes (Lab 3)

In Docker Desktop enable local Kubernetes and ensure the current context is exactly `docker-desktop`. Edit `k8s/deployment.yaml` and replace `YOUR_DOCKERHUB_USERNAME` + `build-1` with your existing public image tag. On Docker Desktop, you can use a public Docker Hub repository; private ones require registry credentials.

```powershell
kubectl config use-context docker-desktop
kubectl config current-context
kubectl get nodes
kubectl apply -f k8s/
kubectl get pods
kubectl get service hotel-cancellation-api
kubectl scale deployment hotel-cancellation-api --replicas=3
kubectl rollout status deployment/hotel-cancellation-api
```

Visit http://localhost:8080/docs on Docker Desktop where LoadBalancer localhost routing is available. Update `image:` to the next real build tag, apply, then show rollback:

```powershell
kubectl apply -f k8s/
kubectl rollout undo deployment/hotel-cancellation-api
kubectl delete -f k8s/
```

## Presentation demo plan (7–10 min)

1. Business problem + Kaggle dataset and schema.
2. Show actual data distribution and class imbalance; explain target `Canceled` (1).
3. Run `python train.py`; compare all 3 candidates + F1 / ROC-AUC in MLflow.
4. Show Git, DVC pointer, private S3 dataset, and `dvc push`.
5. Call `POST /predict` from FastAPI `/docs`; show input validation failing on invalid lead time.
6. Show `pytest` and a successful **GitHub Actions** workflow with quality gate.
7. Show real built Docker Hub image tag and API running inside container.
8. Show Prometheus counters + lead-time drift check, explain alerts and retraining.
9. Optional Kubernetes: replicas, new tag rollout, rollback.

**Important limitations:** No real Kaggle data are bundled in this ZIP; this environment cannot access Kaggle. No training scores, S3 uploads, GitHub commits or Docker Hub releases are claimed as completed. These require your network access and respective accounts. The illustrative sample request uses a historical arrival year consistent with the dataset; avoid interpreting the model as validated for present-day reservations. The local prediction diary is a **single-container learning demo**; production multi-replica logging needs central durable storage, structured events, security, and retention controls.

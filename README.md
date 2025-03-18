# MLOps with MLflow

## 프로젝트 개요
이 프로젝트는 **MLflow를 활용한 MLOps 실험 관리 및 모델 배포**를 다루는 프로젝트입니다.  
MLflow Tracking을 사용하여 실험을 기록하고, Model Registry를 활용하여 최적의 모델을 관리합니다.  

## 📌 실행 방법

### 1️⃣ *MLflow + PostgreSQL 실행**
```bash
docker-compose up -d

source myenv/bin/activate
cd /home/sambong

python3 scripts/mlflow_tracking.py

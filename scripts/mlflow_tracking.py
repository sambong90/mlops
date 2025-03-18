import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("현재 MLflow Tracking URI:", mlflow.get_tracking_uri())
mlflow.set_tracking_uri("http://localhost:5000")
print("현재 MLflow Tracking URI:", mlflow.get_tracking_uri())

# 기존 Experiment가 없을 경우 새로 생성
experiment_name = "iris_classification"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment:
    experiment_id = experiment.experiment_id
else:
    experiment_id = mlflow.create_experiment(experiment_name)

print(f"사용할 Experiment ID: {experiment_id}")

mlflow.set_experiment(experiment_name)  # 실험을 설정할 때 이름이 아니라 ID를 직접 지정 가능

with mlflow.start_run(experiment_id=experiment_id):
    print("MLflow Run Started with Experiment ID:", experiment_id)


# 📌 MLflow 실행 시작
with mlflow.start_run():
    # 데이터셋 로드
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # 모델 및 하이퍼파라미터 설정
    n_estimators = 50
    max_depth = 5
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # 모델 학습
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 정확도 계산
    accuracy = accuracy_score(y_test, y_pred)

    # 📌 MLflow에 실험 기록
    mlflow.log_param("n_estimators", n_estimators)  # 하이퍼파라미터 기록
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)  # 모델 성능 기록
    mlflow.sklearn.log_model(model, "random_forest_model")  # 모델 저장

    print(f"✅ 모델 훈련 완료! 정확도: {accuracy:.4f}")



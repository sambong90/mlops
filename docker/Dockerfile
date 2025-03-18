# 베이스 이미지 (Ubuntu + Python)
FROM python:3.10

# 기본 패키지 설치
RUN apt update && apt install -y curl wget git && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 필수 Python 패키지 설치
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Jupyter Notebook 실행 (포트 8888)
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]


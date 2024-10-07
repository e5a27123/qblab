# 使用 mlflow 官方映像作為基礎映像
FROM ghcr.io/mlflow/mlflow:v2.15.1

# 安裝 psycopg2
RUN pip install psycopg2-binary
# 使用 Python 3.10 Slim 作為基礎映像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 複製當前目錄下的所有文件到容器的 /app 目錄
COPY src .

# 安裝所需的 Python 套件
RUN pip install --no-cache-dir -r /app/requirements-app.txt

# 暴露應用程序運行的端口
ENV PYTHONPATH=/app

RUN chmod -R g=u /app

# 運行 FastAPI 應用
ENTRYPOINT ["sh", "/app/start.sh"]
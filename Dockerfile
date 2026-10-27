FROM python:3.13-slim

WORKDIR /app

# 先装依赖，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 运行时注入 PORT 环境变量即可（默认 8501）
ENV PORT=8501
EXPOSE 8501

CMD ["python", "run_server.py"]

# 使用Python 3.9作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app/

# 安装Python依赖
RUN pip install --no-cache-dir gradio huggingface_hub

# 下载模型资源
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='jzq11111/mooncast', local_dir='./resources/')"

# 暴露端口
EXPOSE 7860

# 启动应用
CMD ["python", "app.py"] 
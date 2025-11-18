FROM m.daocloud.io/docker.io/vllm/vllm-openai:v0.9.1

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    build-essential ninja-build cmake git ccache && \
    rm -rf /var/lib/apt/lists/*

# 设置 CUDA 环境变量（防止找不到 nvcc）
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 复制源码到容器
WORKDIR /workspace
COPY . .

# 清理旧缓存（防止 cutlass/cmake 路径冲突）
RUN rm -rf /workspace/build /workspace/.deps
# 安装必要 Python 构建包
RUN uv pip install --system numpy packaging ninja cmake setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple

ARG HTTPS_PROXY
ARG HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV http_proxy=$HTTP_PROXY

# 安装你自己的 vLLM fork 或扩展
RUN uv pip install --system  -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN uv pip install --system grpcio-tools==1.71.0 protobuf==5.29.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

ENV HTTPS_PROXY=""
ENV HTTP_PROXY=""
ENV NO_PROXY='127.0.0.1,localhost,::1'

ENTRYPOINT []
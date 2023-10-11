FROM python:3

# Update and upgrade the existing packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ninja-build \
        libopenblas-dev \
        build-essential

RUN mkdir /app
WORKDIR /app

# install python dependencies
RUN pip install --upgrade pip
RUN pip install huggingface_hub
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install llama-cpp-python[server]@git+https://github.com/abetlen/llama-cpp-python.git --verbose

# Set environment variable for the host
ENV HOST=0.0.0.0
ENV PORT=8000
ENV HF_REPO=TheBloke/Mistral-7B-OpenOrca-GGUF
ENV HF_FILE=mistral-7b-openorca.Q4_K_M.gguf
ENV MODEL_ALIAS=local-model
ENV CHAT_FORMAT=chatml

# Expose a port for the server
EXPOSE 8000

COPY run.sh /app
# Run the server start script
CMD ["/bin/sh", "/app/run.sh"]

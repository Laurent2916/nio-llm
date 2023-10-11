FROM python:3.9

# Update and upgrade the existing packages
RUN apt-get update && \
    apt-get upgrade -y

RUN mkdir /app
WORKDIR /app

# install python dependencies
RUN pip install --upgrade pip
RUN pip install \
    jsonargparse[signatures] \
    git+https://github.com/abetlen/llama-cpp-python.git \
    matrix-nio \
    openai \
    rich

COPY *.py /app/nio_llm/
ENV PYTHONPATH=/app
CMD ["/usr/local/bin/python", "/app/nio_llm"]

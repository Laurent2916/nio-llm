#!/bin/bash

huggingface-cli download $HF_REPO $HF_FILE
MODEL_PATH=`huggingface-cli download --quiet $HF_REPO $HF_FILE`

python3 -m llama_cpp.server --host $HOST --port $PORT --model $MODEL_PATH --model_alias $MODEL_ALIAS --chat_format $CHAT_FORMAT

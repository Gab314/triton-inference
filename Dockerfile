FROM nvcr.io/nvidia/tritonserver:23.11-py3
RUN  python3 -m pip install --upgrade pip
COPY hf_model /models
RUN pip install torch transformers
EXPOSE 8000 8001 8002
CMD ["tritonserver", "--model-repository=/models"]
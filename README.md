# triton-inference

Example structure to deploy Nvidia's Triton Inference in a Kubernetes Cluster. Including multi-model inference with parallelization.



## Step-by-step:

1. **Install requirements**:

Default
```bash
    pip install -r requirements.txt
```

Azure ML
```bash
    pip install -r requirements_az.txt
```

2. **Load model:**

+ a. `cd hf_model`
+ b. `mkdir {model_name}`
+ c. Update `name` inside `config.pbtxt`

```txt
# config.pbtxt
name: "{model_name}"
```

+ d. `mkdir {model_version}`
+ e. Copy the model files, e.g. `model.pt` 
+ f. For Custom Models, e.g. HuggingFace, update `/1/model.py`


3. **Run docker:**

```bash
docker run --gpus all -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/hf_model:/models nvcr.io/nvidia/tritonserver:23.11-py3 tritonserver --model-repository=/models
```
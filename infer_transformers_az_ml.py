# # Test the blue deployment with some sample data
import requests
import gevent.ssl
import numpy as np
import tritonclient.http as tritonhttpclient
from pathlib import Path
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from PIL import Image
from torchvision import transforms
import requests
from datetime import datetime
import multiprocessing
import statistics
from tritonclient.utils import np_to_triton_dtype

subscription_id = ""
resource_group = ""
workspace_name = ""


def azure_inference(data):

    inputs, outputs = data
    init = datetime.utcnow()
    try:
        # Querying the server
        results = triton_client.async_infer(model_name, inputs=[inputs], outputs=[outputs], headers=headers)

    except Exception as e:
        return "ERROR"
    
    return (datetime.utcnow() - init).total_seconds()


def multi_inference(inputs, outputs, processes, task_size):


    pool = multiprocessing.Pool(processes=processes)
    results = pool.map(azure_inference, [(inputs, outputs)] * task_size)

    return results


if __name__ == '__main__':

    ENDPOINT_NAME = "gabriel-sozhf"
    MODEL_DEPLOYMENT = ""
    MODEL_NAME = ""
    AUTH_KEY = ""
    TASK_SIZE = 20
    
    # Setting up client
    client = MLClient(
        DefaultAzureCredential(),
        subscription_id,
        resource_group,
        workspace_name,
    )

    endpoint = client.online_endpoints.get(ENDPOINT_NAME)
    scoring_uri = endpoint.scoring_uri
    # We remove the scheme from the url
    url = scoring_uri[7:]

    # Initialize client handler
    triton_client = tritonhttpclient.InferenceServerClient(
        url=url
    )

    # Create headers
    headers = {}
    auth_key = AUTH_KEY
    headers["Authorization"] = f"Bearer {auth_key}"
    headers['Content-Type'] = 'application/json'
    headers['azureml-model-deployment'] = MODEL_DEPLOYMENT

    # Check status of triton server
    health_ctx = triton_client.is_server_ready(headers=headers)
    print("Is server ready - {}".format(health_ctx))

    # # Check status of model
    model_name = MODEL_NAME
    status_ctx = triton_client.is_model_ready(model_name, "1", headers)
    print("Is model ready - {}".format(status_ctx))


    test_sentence = ['Isto um teste']
    sentences = np.array(test_sentence, dtype=object)
    inputs = tritonhttpclient.InferInput("sentence", sentences.shape, np_to_triton_dtype(sentences.dtype))
    
    inputs.set_data_from_numpy(sentences, binary_data=True)

    outputs = tritonhttpclient.InferRequestedOutput(
        "label", binary_data=True, class_count=0
    )

    avgs = []
    for processes in range(1, 10):

        print(f"---------------------------------------------------")
        print(f"Simultaneous Processes: {processes}")
        total_mean_times = []

        init = datetime.utcnow()
        for i in range(5):
            response = multi_inference(inputs, outputs, processes, TASK_SIZE)
            errors = [r for r in response if r == "ERROR"]

            if len(errors) != len(response):
                mean_time = statistics.mean([r for r in response if r != "ERROR"])
                print(f"Mean time per task: {mean_time}, Errors: {len(errors)}")
                if i != 0: total_mean_times.append(mean_time)
            else:
                print("Skip: Multiple errors")
        
        time_taken = (datetime.utcnow() - init).total_seconds()
        print(f"AVG time per 5 inference:  {time_taken}")
        total_mean_time = statistics.mean(total_mean_times)
        avgs.append(total_mean_time)
    
    print(f"List of AVGs: {avgs}")

# # Test the blue deployment with some sample data
import requests
import gevent.ssl
import numpy as np
import tritonclient.http as tritonhttpclient
from pathlib import Path
import requests
from datetime import datetime
import multiprocessing
import statistics
from tritonclient.utils import np_to_triton_dtype
import tritonclient.http as httpclient

def azure_inference(data):

    inputs, outputs = data
    init = datetime.utcnow()
    try:
        # Querying the server
        results = client.async_infer("transformers", inputs=[inputs], outputs=[outputs])

    except Exception as e:
        return "ERROR"
    
    return (datetime.utcnow() - init).total_seconds()




def multi_inference(inputs, outputs, processes, task_size):


    pool = multiprocessing.Pool(processes=processes)
    results = pool.map(azure_inference, [(inputs, outputs)] * task_size)

    return results


if __name__ == '__main__':
    TASK_SIZE = 20
    test_sentence = ['Isto um teste']
    sentences = np.array(test_sentence, dtype=object)
    inputs = httpclient.InferInput("sentence", sentences.shape, np_to_triton_dtype(sentences.dtype))

    inputs.set_data_from_numpy(sentences, binary_data=True)

    # Setting up client
    client = httpclient.InferenceServerClient(url="localhost:8000")

    outputs = httpclient.InferRequestedOutput(
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

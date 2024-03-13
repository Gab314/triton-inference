

import numpy as np
import tritonclient.http as httpclient
import numpy as np
import json
from tritonclient.utils import np_to_triton_dtype

test_sentence = ['Isto um teste']
sentences = np.array(test_sentence, dtype=object)
inputs = httpclient.InferInput("sentence", sentences.shape, np_to_triton_dtype(sentences.dtype))

inputs.set_data_from_numpy(sentences, binary_data=True)

parameters = {
    'token' : "1234"
}

# Setting up client
client = httpclient.InferenceServerClient(url="localhost:8000")

outputs = httpclient.InferRequestedOutput(
    "label", binary_data=True, class_count=0
)


results = client.infer(model_name="transformers", inputs=[inputs], outputs=[outputs], parameters=parameters)
inference_output = results.as_numpy("label").tolist()

output = json.loads(inference_output)
print("class", output['class'])
print("prob:", output['prob'])
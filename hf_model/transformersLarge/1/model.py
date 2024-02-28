
import io
import json
import math
import logging
import numpy as np
import os
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
import subprocess
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        global device

        # print(f"Staring at: {os.getcwd()}")
        # subprocess.run(["pip", "install", "transformers==4.37.2", "torch==2.1.2"]) 

        # from transformers import BertTokenizerFast, BertForSequenceClassification
        # import torch
        # AZUREML_MODEL_DIR is an environment variable created during deployment.
        # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
        # Please provide your model's folder name if there is one
        # local_directory = '/var/azureml-app/azureml-models/hfmodel/1/hf_model/transformers'
        local_directory = '/models/transformersLarge'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizerFast.from_pretrained(local_directory)
        self.model = BertForSequenceClassification.from_pretrained(local_directory)
        self.model.eval()
        self.model.to(device)
        
        logging.info("Segmenter init complete")


    def execute(self, requests):
        responses = []
        for sentence in requests:
            inp = pb_utils.get_input_tensor_by_name(sentence, "sentence").as_numpy()[0].decode("utf-8")

            tokenized = self.tokenizer(
                inp, 
                padding='max_length', 
                truncation=True, 
                max_length=192,
                return_tensors='pt'
            )
            
            out = self.model(
                tokenized['input_ids'].to(device),
                attention_mask=tokenized['attention_mask'].to(device)
                )

            result = torch.max(torch.softmax(out.logits, dim=1), dim=1)
            probs = result.values.detach().cpu().numpy()

            # Sending results
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "label",
                    probs
                )
            ])

            responses.append(inference_response)

        return responses
    

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")

import os
import torch
import argparse
import numpy as np
import pandas as pd
from config import *
from utils import model_loading, load_messages, batch_inference_fn
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--batch", dest="batch_size", type=int, default=3, help="batch size for inference on the model.")
parser.add_argument("--name", dest="model_name", type=str, default="sentimental_camembert_model.pth",
                    help="the name of the model to use for inference.")
parser.add_argument("--repository", dest="models_repository", type=str, default="saved",
                    help="repository where the model to use for inference is.")
parser.add_argument("--data-path", type=str, help="data that we want to use for the inference will be stored.",
                    default='daily_batches/cine_opke_oluwa_Jan-21-2021.parquet')
parser.add_argument("--workers", dest="num_workers", type=int, default=6, 
                    help="number of processors to transform the data.")
parser.add_argument("--save-path", dest="save_path", type=str, default="predictions", 
                    help="where the results of the inference will be stored.")
parser.add_argument("--device", type=str, default="cuda:0", 
                    help="determine which gpu to use (or cpu) to predict on the model.")

# Taking back the variables from the parser
inference_args = parser.parse_args()
batch_size = inference_args.batch_size
model_name = inference_args.model_name
models_repository = inference_args.models_repository
num_workers = inference_args.num_workers
save_path = inference_args.save_path
data_path = inference_args.data_path
device = inference_args.device

model_path = os.path.join(models_repository, model_name)

# Load the model in memory 
model = model_loading(model_path) 
model.to(device)
                                  
# Make the batch-inference
batch_inference_fn(model, batch_size, num_workers, max_length, data_path, save_path, device)
print(" ======== End of batch predictions ! ========")
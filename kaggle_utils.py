from kaggle.api.kaggle_api_extended import KaggleApi
import os
import json
import torch
from tqdm import tqdm
from typing import List, Optional, Union
import pandas as pd
from sklearn.metrics import average_precision_score
import shutil
import subprocess
from pathlib import Path
kaggle_api_token = {
    "username": "inayarahmanisa",
    "key": "***REMOVED***"
}

os.environ['KAGGLE_USERNAME'] = kaggle_api_token['username']
os.environ['KAGGLE_KEY'] = kaggle_api_token['key']

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

def download_from_kaggle(
        dataset_name:str,
        file_name:str,
        download_path: str = None

        ):
    """
    dataset_name: is username/dataset_name like 'inayarahmanisa/testssssss'
    """
    download_path = download_path if download_path else "data/"
    api.dataset_download_file(dataset_name, file_name, path=download_path)
    # api.dataset_download_files(dataset_name, path=download_path if download_path else None, unzip=True)

    print("Dataset downloaded and unzipped.")



def save_to_kaggle(
    dataset_name: str,
    data_dir:str = None,
    result_neurons: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    filename: Optional[Union[str, List[str]]] = None,
    is_update:bool =False
):
    
    os.environ['KAGGLE_USERNAME'] = kaggle_api_token['username']
    os.environ['KAGGLE_KEY'] = kaggle_api_token['key']

    api = KaggleApi()
    api.authenticate()

    target_dir = "kaggle/output"
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    os.makedirs(target_dir, exist_ok=True)

    base_dir = os.path.join(target_dir, "result")
    
    os.makedirs(base_dir, exist_ok=True)

    meta = {
        "id": f"{kaggle_api_token['username']}/{dataset_name}",
        "title": f"{dataset_name}",
        "isPrivate": True,
        "licenses": [{"name": "other"}]
    }

    metadata_path = os.path.join(base_dir, 'dataset-metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(meta, f)
    if data_dir:
        shutil.copytree(data_dir, base_dir, dirs_exist_ok=True)
    elif result_neurons:
        if isinstance(result_neurons, torch.Tensor):
            assert isinstance(filename, str), "there must be the same number of filename and rensult neurons"
            torch.save(result_neurons, os.path.join(base_dir, filename))
            # Process single tensor
        elif isinstance(result_neurons, list) and all(isinstance(t, torch.Tensor) for t in result_neurons):
            assert len(result_neurons) == len(filename), "there must be the same number of filename and rensult neurons"
            for i in range (len(result_neurons)):
                torch.save(result_neurons[i], os.path.join(base_dir, filename[i]))

    print("Files in the target directory:")
    for file_name in os.listdir(base_dir):
        print(f"- {file_name}")

    base_dir = Path("kaggle/output/result").resolve()
    save_dir = base_dir.as_posix() 
    print(f"save_dir: {save_dir}")
    try:
        dataset_ref = f"{kaggle_api_token['username']}/{dataset_name}"
        if is_update:
            print(f"Dataset {dataset_name} exists. Updating...")
            api.dataset_download_files(dataset_ref, path=save_dir, unzip=True)
            subprocess.run(["kaggle", "datasets", "version", "-p", base_dir, "-m", "Updating dataset"])

        else:
            print(f"Dataset {dataset_name} does not exist. Creating a new one...")
            subprocess.run(["kaggle", "datasets", "create", "-p", base_dir, "--dir-mode", "zip"])

    except Exception as e:
        print(f"Error while checking or creating dataset: {e}")

    shutil.rmtree(target_dir)
# c = torch.randn(1,2,3,4)
# save_to_kaggle(
#     result_neurons = [c, c],
#     dataset_name= "testssssss",
#     filename= ["test.pt", "test2.pt"],
#     is_update=True
# )

# download_from_kaggle('inayarahmanisa/testssssss', 'test.pt')
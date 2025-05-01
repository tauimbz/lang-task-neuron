import os
import json
import subprocess

import os
import json
import kaggle_utils

# Set the KAGGLE_CONFIG_DIR environment variable if the kaggle.json is not in the default location
os.environ['KAGGLE_CONFIG_DIR'] = '/path/to/your/kaggle.json'  # Update this with the actual path

# Define the dataset you want to download
dataset_name = 'nayechz/neuron-xwinograd-csv'  # Replace with your dataset's name
file_name = 'dict_raw_gemma2_xwinograd_all.pt'

# Download the dataset using Kaggle API
kaggle.api.dataset_download_files(dataset_name, path='./', unzip=True)

# Now the dataset should be downloaded in your current working directory, 
# and the file 'dict_raw_gemma2_xwinograd_all.pt' should be in the 'neuron-xwinograd-csv' folder


# # # Save kaggle.json credentials inline
# kaggle_api_token = {
#     "username": "inayarahmanisa",
#     "key": "***REMOVED***"
# }

# # # Make the .kaggle directory if it doesn't exist
# # os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

# # # Write the API token into the kaggle.json file
# # with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
# #     json.dump(kaggle_api_token, f)

# # # Secure the kaggle.json file by changing its permissions
# # os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

# # subprocess.run(['kaggle', 'datasets', 'download', '-d', 'nayechz/neuron-xwinograd-csv'])


# import os
# import json
# import torch
# import shutil  # Import shutil for recursive removal

# # from kaggle_secrets import UserSecretsClient

# # Retrieve the Kaggle API credentials using Kaggle secrets
# # user_secrets = UserSecretsClient()
# os.environ['KAGGLE_KEY'] = kaggle_api_token['key']
# os.environ['KAGGLE_USERNAME'] = kaggle_api_token['username']

# # Clear the target directory if it exists (Equivalent to 'rm -rf')
# target_dir = "/kaggle/dataset/dist_method_result"
# if os.path.exists(target_dir):
#     shutil.rmtree(target_dir)   # You may need to handle directory removal recursively depending on its content.

# # Recreate necessary directories
# os.makedirs(target_dir, exist_ok=True)

# # Set the base directory for saving data
# base_dir = os.path.join(target_dir, "dist_method_result")
# os.makedirs(base_dir, exist_ok=True)

# # Optionally, create subdirectories if needed
# # os.makedirs(os.path.join(base_dir, "kde"), exist_ok=True)
# # os.makedirs(os.path.join(base_dir, "qpdf"), exist_ok=True)
# # os.makedirs(os.path.join(base_dir, "kl_results"), exist_ok=True)
# # os.makedirs(os.path.join(base_dir, "js_results"), exist_ok=True)

# dataset_name = "testssssss"
# # Create metadata for the dataset
# meta = {
#     "id": f"inayarahmanisa/{dataset_name}",  # Change the ID as per your dataset specifics
#     "title": f"{dataset_name}",  # Dataset title
#     "isPrivate": True,  # Dataset privacy
#     "licenses": [{"name": "other"}]  # Dataset license
# }

# # Write metadata to a JSON file
# metadata_path = os.path.join(base_dir, 'dataset-metadata.json')
# with open(metadata_path, 'w') as f:
#     json.dump(meta, f)

# # Save the tensor data (map30, map50, map80, map90, map99) to disk
# # torch.save(map30, os.path.join(base_dir, "map30.pt"))
# # torch.save(map50, os.path.join(base_dir, "map50.pt"))
# # torch.save(map80, os.path.join(base_dir, "map80.pt"))
# # torch.save(map90, os.path.join(base_dir, "map90.pt"))
# import torch 
# c = torch.randn(2,4)
# torch.save(c, os.path.join(base_dir, "map99.pt"))



# # List the files in the target directory (Equivalent to 'ls -h')
# print("Files in the target directory:")
# for filename in os.listdir(base_dir):
#     print(f"- {filename}")

# # Now you can create the Kaggle dataset (by calling Kaggle API or using CLI via subprocess)
# # If you have Kaggle CLI set up, you could use subprocess to run the command (Kaggle dataset create).
# import subprocess
# subprocess.run(["kaggle", "datasets", "create", "-p", base_dir, "--dir-mode", "zip"])

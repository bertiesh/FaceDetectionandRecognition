import argparse

from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.models import BatchDirectoryInput, Input

# Define the URL and set up client
BULK_UPLOAD_MODEL_URL = "http://127.0.0.1:5000/bulkupload"
client = MLClient(BULK_UPLOAD_MODEL_URL)

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="To parse text arguments")
parser.add_argument(
    "directory_paths",
    metavar="file",
    type=str,
    nargs="+",
    help="Path to directory containing images",
)
args = parser.parse_args()

parameters = {}
inputs = {
    "directory_paths": Input(
        root=BatchDirectoryInput.model_validate(
            {
                "directories": [
                    {"path": directory_path} for directory_path in args.directory_paths
                ]
            }
        )
    )
}

response = client.request(inputs, parameters)
print("Find Face model response")
print(response, "\n")

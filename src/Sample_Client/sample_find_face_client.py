import argparse

from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.models import BatchFileInput, Input

# Define the URL and set up client
IMAGE_MATCH_MODEL_URL = "http://127.0.0.1:5000/findface"
client = MLClient(IMAGE_MATCH_MODEL_URL)

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="To parse text arguments")
parser.add_argument(
    "--file_paths", metavar="file", type=str, nargs="+", help="Path to images"
)

parser.add_argument(
    "--database_path", required=True, type=str, help="Path to the database file"
)

args = parser.parse_args()

parameters = {"database_path": args.database_path}
inputs = {
    "image_paths": Input(
        root=BatchFileInput.model_validate(
            {"files": [{"path": file_path} for file_path in args.file_paths]}
        )
    )
}

response = client.request(inputs, parameters)
answer = ""

# Return all similar images as space separated file_names
for file in response["files"]:
    answer += file["path"].split("\\")[-1] + " "
print(answer, "\n")

import argparse
import os

from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.models import BatchFileInput, Input

from src.facematch.utils.resource_path import get_resource_path

# Define the URL and set up client
IMAGE_MATCH_MODEL_URL = "http://127.0.0.1:5000/findface"
client = MLClient(IMAGE_MATCH_MODEL_URL)

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="To parse text arguments")

# Absolute path of query image
parser.add_argument(
    "--file_paths", metavar="file", type=str, nargs="+", help="Path to images"
)
# Name of database from user
parser.add_argument(
    "--database_name", required=True, type=str, help="Name of the database file"
)

# Face Similarity threshold from user
parser.add_argument(
    "--similarity_threshold",
    type=float,
    help="Return matches with similarity above this threshold",
)

args = parser.parse_args()

# Check if database exists
# if not os.path.exists(
#     get_resource_path(os.path.join("data", args.database_name + ".csv"))
# ):
#     print("Database does not exist")
#     exit()

# Set parameters and inputs for the request
parameters = {
    "similarity_threshold": args.similarity_threshold,
    "database_name": args.database_name,
}
inputs = {
    "image_paths": Input(
        root=BatchFileInput.model_validate(
            {"files": [{"path": file_path} for file_path in args.file_paths]}
        )
    )
}

# Response from server
response = client.request(inputs, parameters)

# Show results to user

# If reponse is a TextResponse, print the value
if response["output_type"] == "text":
    print(response["value"])

else:
    # Show matches found
    print("\nMatches found")
    for file in response["files"]:
        print(file["path"].split("\\")[-1])

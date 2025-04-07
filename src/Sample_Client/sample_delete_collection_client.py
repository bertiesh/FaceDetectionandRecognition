import argparse
from dotenv import load_dotenv
import json

from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.models import TextInput, Input

from src.facematch.utils.resource_path import get_config_path

load_dotenv()

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="To parse text arguments")

# Name of embedding collection from user
parser.add_argument(
    "--collection_name", required=True, type=str, help="Name of the collection"
)

# Name of embedding collection from user
parser.add_argument(
    "--model_name", required=False, type=str, help="Name of the model"
)

args = parser.parse_args()

if not args.model_name:
# Get models from config file.
    config_path = get_config_path("model_config.json")
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    model_name = config["model_name"]
else:
    model_name = args.model_name
    
# Define the URL and set up client
IMAGE_MATCH_MODEL_URL = "http://127.0.0.1:5000/deletecollection"
client = MLClient(IMAGE_MATCH_MODEL_URL)


inputs = {
    "collection_name": Input(
        root=TextInput.model_validate(
            {
                "text": args.collection_name
            }
        )
    ),
    "model_name": Input(
        root=TextInput.model_validate(
            {
                "text": model_name
            }
        )
    )
}

# Response from server
response = client.request(inputs, {})
print(response)
print(response["value"])
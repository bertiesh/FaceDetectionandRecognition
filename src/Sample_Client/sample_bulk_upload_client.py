import argparse
import os

from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.models import BatchDirectoryInput, Input

from src.facematch.utils.resource_path import get_resource_path

# Define the URL and set up client
BULK_UPLOAD_MODEL_URL = "http://127.0.0.1:5000/bulkupload"
client = MLClient(BULK_UPLOAD_MODEL_URL)

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="To parse text arguments")

# Absolute path of directory of images
parser.add_argument(
    "--directory_paths",
    metavar="file",
    type=str,
    nargs="+",
    help="Path to directory containing images",
)

# Name of database from user
parser.add_argument(
    "--database_name", required=True, type=str, help="Name of the database file"
)

args = parser.parse_args()

# Dropdown database path is used to give the option of creating a new database and selecting an existing database for users in frontend
# Set dropdown database path to the name of the database if it exists, otherwise set it to "Create a new database"
if os.path.exists(get_resource_path((args.database_name + ".csv"))):
    dropdown_database_name = args.database_name
else:
    dropdown_database_name = "Create a new database"

# Set parameters and inputs for the request
parameters = {
    "database_name": args.database_name,
    "dropdown_database_name": dropdown_database_name,
}
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

# Response from the server
response = client.request(inputs, parameters)
print("Bulk Upload model response")
print(response, "\n")

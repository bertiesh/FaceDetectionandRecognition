import argparse
import os
import time
from dotenv import load_dotenv
import chromadb
import ast

from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.models import DirectoryInput, Input

load_dotenv()

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="To parse text arguments")

# Absolute path of query image
parser.add_argument(
    "--query_directory", required=True, metavar="directory", type=str, help="Path to images"
)

# Name of embedding collection from user
parser.add_argument(
    "--collection_name", required=True, type=str, help="Name of the collection file"
)

# Face Similarity threshold from user
parser.add_argument(
    "--similarity_threshold",
    type=float,
    help="Return matches with similarity above this threshold",
)

args = parser.parse_args()

# Define the URL and set up client
IMAGE_MATCH_MODEL_URL = "http://127.0.0.1:5000/findfacebulk"
client = MLClient(IMAGE_MATCH_MODEL_URL)

DBclient = chromadb.HttpClient(host='localhost', port=8000)

# Check if collection exists
collections = DBclient.list_collections()

if args.collection_name not in map(lambda c: c.split("_")[0],collections):
    print("Collection does not exist")
    exit()

# Set parameters and inputs for the request
parameters = {
    "similarity_threshold": args.similarity_threshold,
    "collection_name": args.collection_name,
}

absolute_query_directory = os.path.abspath(args.query_directory)

inputs = {
    "query_directory": Input(
        root=DirectoryInput.model_validate(
            {
                "path": absolute_query_directory
            }
        )
    )
}

# Response from server
start_time = time.time()
response = client.request(inputs, parameters)
try:
    results = ast.literal_eval(response['value'])
    query_paths = os.listdir(args.query_directory)
    query_paths.sort()
    for match_paths, query_path in zip(results, query_paths):
        query_name = os.path.basename(query_path)
        match_paths = " ".join(list(map(lambda path: os.path.basename(path),match_paths)))
        print(f"Query: {query_name}    Matches: {match_paths}")

except Exception:
    # If response is not an array, print the value
    print(response["value"])


end_time = time.time()

elapsed_time = end_time - start_time
print(f"Time taken for face find bulk: {elapsed_time}")
import argparse
import csv
import os
import time
from dotenv import load_dotenv
import chromadb
import ast

from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.models import DirectoryInput, Input

def query_find_face_bulk(query_directory, collection_name, similarity_threshold):
    # Define the URL and set up client
    IMAGE_MATCH_MODEL_URL = "http://127.0.0.1:5000/findfacebulk"
    client = MLClient(IMAGE_MATCH_MODEL_URL)

    DBclient = chromadb.HttpClient(host='localhost', port=8000)

    # Check if collection exists
    collections = DBclient.list_collections()

    if collection_name not in map(lambda c: c.split("_")[0],collections):
        print("Collection does not exist")
        exit()

    # Set parameters and inputs for the request
    parameters = {
        "similarity_threshold": similarity_threshold,
        "collection_name": collection_name,
    }

    absolute_query_directory = os.path.abspath(query_directory)

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
    response = client.request(inputs, parameters)
    try:
        response_data = ast.literal_eval(response['value'])
        return response_data
    
    except:
        # If response is not an array, print the value
        print(response["value"])
        return response["value"]


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

# Name of model from user
parser.add_argument(
    "--model_name", required=True, type=str, help="Name of the model file"
)

args = parser.parse_args()

similarity_threshold = 0.501


start_time = time.time()

results = query_find_face_bulk(args.query_directory, args.collection_name, similarity_threshold)

output_csv_path = f'{os.getenv("OUTPUT_CSV_PATH")}_{args.model_name}_{similarity_threshold}.csv'

with open(output_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['filename', 'result'])

query_paths = os.listdir(args.query_directory)
query_paths.sort()
for match_paths, query_path in zip(results, query_paths[99:601]):
    query_name = os.path.basename(query_path)
    match_paths = " ".join(list(map(lambda path: os.path.basename(path),match_paths)))
    with open(output_csv_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([query_name, match_paths])

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Time taken for face find bulk: {elapsed_time}")
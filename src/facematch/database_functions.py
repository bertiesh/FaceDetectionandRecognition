import os

import numpy as np
import pandas as pd

from src.facematch.FAISS import add_embeddings_faiss_index

import chromadb

client = chromadb.HttpClient(host='localhost', port=8000)

def upload_embedding_to_database(data, database_filepath):
    df = pd.DataFrame(data)
    # df["embedding"] = df["embedding"].apply(lambda x: ",".join(map(str, x)))
    df["bbox"] = df["bbox"].apply(lambda x: ",".join(map(str, x)))

    model_name = df["model_name"].iloc[0]

    # embeddings_array = np.array([d["embedding"] for d in data])
    metadatas = [{"image_path":d["image_path"]} for d in data]

    collection = client.get_or_create_collection(name=f"{model_name.lower()}_collection", metadata={
        "image_path": "Original path of the uploaded image"
    })

    collection.add(
        embeddings=list(df['embedding']),
        metadatas=metadatas,
        ids=list(df["sha256_image"]),
    )

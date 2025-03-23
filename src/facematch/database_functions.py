import os

import numpy as np
import pandas as pd

from src.facematch.FAISS import add_embeddings_faiss_index
from sklearn.metrics.pairwise import cosine_similarity

import chromadb

client = chromadb.HttpClient(host='localhost', port=8000)

def get_collection(model_name, client):
    return client.get_or_create_collection(name=f"{model_name.lower()}_collection", metadata={
        "image_path": "Original path of the uploaded image",
        "hnsw:space": "l2",
    })

def upload_embedding_to_database(data, database_filepath):
    df = pd.DataFrame(data)
    # df["embedding"] = df["embedding"].apply(lambda x: ",".join(map(str, x)))
    df["bbox"] = df["bbox"].apply(lambda x: ",".join(map(str, x)))

    model_name = df["model_name"].iloc[0]

    # embeddings_array = np.array([d["embedding"] for d in data])
    metadatas = [{"image_path":d["image_path"]} for d in data]

    collection = get_collection(model_name, client)

    collection.add(
        embeddings=list(df['embedding']),
        metadatas=metadatas,
        ids=list(df["sha256_image"]),
    )

def query(data, n_results, threshold):
    query_vector = np.array(data["embedding"])
    collection = get_collection(data["model_name"], client)

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["metadatas", "distances", "embeddings"]
    )

    import pdb; pdb.set_trace()

    paths = list(map(lambda x: x["image_path"],result["metadatas"][0]))
    results = pd.DataFrame({"embedding":list(result["embeddings"][0]), "distance":result["distances"][0], "img_path":paths})


    get_similarity = lambda query_vector: lambda x: cosine_similarity(x.reshape(1, -1), query_vector.reshape(1, -1))[0][0]

    results["similarity"] = results["embedding"].apply(get_similarity(query_vector))

    if threshold is not None:
        # Filter the DataFrame based on the threshold
        results = results[results["similarity"] >= threshold]
    
    # sort results by similarity in descending order
    results = results.sort_values(by="similarity", ascending=False)

    top_img_paths = results["img_path"].to_list()

    return top_img_paths
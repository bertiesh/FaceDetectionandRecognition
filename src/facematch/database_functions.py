import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

import chromadb

client = chromadb.HttpClient(host='localhost', port=8000)

def get_collection(collection, model_name, client):
    return client.get_or_create_collection(name=f"{collection}_{model_name.lower()}_collection", metadata={
        "image_path": "Original path of the uploaded image",
        "hnsw:space": "l2",
    })

def upload_embedding_to_database(data, collection):
    df = pd.DataFrame(data)
    df["bbox"] = df["bbox"].apply(lambda x: ",".join(map(str, x)))

    model_name = df["model_name"].iloc[0]

    metadatas = [{"image_path":d["image_path"]} for d in data]

    collection = get_collection(collection, model_name, client)

    collection.add(
        embeddings=list(df['embedding']),
        metadatas=metadatas,
        ids=list(df["sha256_image"]),
    )

def query(collection, data, n_results, threshold):
    query_vector = np.array(data["embedding"])
    collection = get_collection(collection, data["model_name"], client)

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["metadatas", "distances", "embeddings"]
    )

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
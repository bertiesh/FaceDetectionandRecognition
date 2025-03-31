import pandas as pd
import numpy as np

import chromadb

client = chromadb.HttpClient(host='localhost', port=8000)

def get_collection(collection, model_name, client):
    return client.get_or_create_collection(name=f"{collection}_{model_name.lower()}", metadata={
        "image_path": "Original path of the uploaded image",
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 750,
        "hnsw:search_ef": 750,
        "hnsw:M": 256
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
    query_vectors = [image["embedding"] for image in data]
    collection = get_collection(collection, data[0]["model_name"], client)
    
    result = collection.query(
        query_embeddings=query_vectors,
        n_results=n_results,
        include=["metadatas", "distances", "embeddings"]
    )

    # Flatten results and include index
    data = []
    for idx, (ids, distances, embeddings, metadatas) in enumerate(zip(result["ids"], result["distances"], result["embeddings"], result["metadatas"])):
        for (image_id, distance, embedding, metadata) in zip(ids, distances, embeddings, metadatas):
            data.append({
                "query_index": idx,  # Index of the original face in the query
                "id": image_id,
                "distance": distance,
                "embedding": embedding.tolist(),
                "img_path": metadata["image_path"]
            })

    # Convert to DataFrame
    result_df = pd.DataFrame(data)
        
    result_df["similarity"] = 1 - result_df["distance"]

    

    if threshold is not None:
        # Filter the DataFrame based on the threshold
        result_df = result_df[result_df["similarity"] >= threshold]
    
    # sort results by similarity in descending order
    result_df = result_df.sort_values(by=["query_index", "similarity"], ascending=[True, False])

    top_img_paths = result_df["img_path"].to_list()

    return top_img_paths


def query_bulk(collection, data, n_results, threshold):
    vectors_per_query = np.array(list(map(lambda query: len(query), data)))
    vectors_per_query_idx = np.cumsum(vectors_per_query)[:-1]

    query_vectors = [face["embedding"] for query in data for face in query]

    collection = get_collection(collection, data[0][0]["model_name"], client)

    result = collection.query(
        query_embeddings=query_vectors,
        n_results=n_results,
        include=["metadatas", "distances", "embeddings"]
    )

    for param in ["ids", "distances", "embeddings", "metadatas"]:
        result[param] = np.split(result[param], vectors_per_query_idx)

    # Flatten results and include index
    data = []
    for query_idx, (q_ids, q_distances, q_embeddings, q_metadatas) in enumerate(zip(result["ids"], result["distances"], result["embeddings"], result["metadatas"])):
        if len(q_ids) == 0:  # If the query has no results, insert a placeholder
            data.append({
                "query_index": query_idx,
                "face_idx": 0,
                "id": None,
                "distance": 1,
                "embedding": None,
                "img_path": None
            })
            continue
        for face_idx, (f_ids, f_distances, f_embeddings, f_metadatas) in enumerate(zip(q_ids, q_distances, q_embeddings, q_metadatas)):
            for (image_id, distance, embedding, metadata) in zip(f_ids, f_distances, f_embeddings, f_metadatas):
                data.append({
                    "query_index": query_idx,
                    "face_idx": face_idx,
                    "id": image_id,
                    "distance": distance,
                    "embedding": embedding.tolist(),
                    "img_path": metadata["image_path"]
                })

    # Convert to DataFrame
    result_df = pd.DataFrame(data)
        
    result_df["similarity"] = 1 - result_df["distance"]

    # sort results by similarity in descending order
    result_df = result_df.sort_values(by=["query_index", "face_idx", "similarity"], ascending=[True, True, False])

    # Function to filter paths based on similarity threshold, but keep an empty list if none qualify
    def extract_paths(group):
        paths = group.loc[group['similarity'] >= threshold, 'img_path'].tolist()
        return paths if paths else []

    # Group by 'index' and extract paths while preserving order
    top_img_paths = result_df.groupby('query_index', sort=False).apply(extract_paths).tolist()

    return top_img_paths
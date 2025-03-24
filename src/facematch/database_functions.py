import pandas as pd

import chromadb

client = chromadb.HttpClient(host='localhost', port=8000)

def get_collection(collection, model_name, client):
    return client.get_or_create_collection(name=f"{collection}_{model_name.lower()}", metadata={
        "image_path": "Original path of the uploaded image",
        "hnsw:space": "cosine",
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
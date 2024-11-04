from deepface import DeepFace


# Function that takes in path to image and returns a status field and a list of face embeddings and corresponding
# region for all faces in the image.
def detect_faces_and_get_embeddings(image_path, model_name, detector_backend):
    try:
        results = DeepFace.represent(
            image_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True,
        )
        face_embeddings = []

        for i, result in enumerate(results):
            embedding = result["embedding"]

            x, y, width, height = (
                result["facial_area"]["x"],
                result["facial_area"]["y"],
                result["facial_area"]["w"],
                result["facial_area"]["h"],
            )

            face_embeddings.append(
                {
                    "image_path": image_path,
                    "embedding": embedding,
                    "bbox": [x, y, width, height],
                }
            )

        return True, face_embeddings
    except Exception as e:
        return False, [e]

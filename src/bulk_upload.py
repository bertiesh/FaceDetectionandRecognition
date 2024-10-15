import os

import face_representation as fr


# Function that takes in path to directory of images to upload to database and returns a success or failure message.
def bulk_upload(image_directory_path):
    try:
        for filename in os.listdir(image_directory_path):
            image_path = os.path.join(image_directory_path, filename)
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                embedding_output = fr.detect_faces_and_get_embeddings(image_path)
                print(embedding_output)
                pass  # call function to upload embeddings to database
        return "Successfully uploaded to database"
    except Exception as e:
        return f"An error occurred: {str(e)}"

import json
import os

from src.facematch.database_functions import upload_embedding_to_database
from src.facematch.face_representation import detect_faces_and_get_embeddings
from src.facematch.logger import log_info
from src.facematch.resource_path import get_resource_path
from src.facematch.similarity_search import (cosine_similarity_search,
                                             cosine_similarity_search_faiss)


class FaceMatchModel:
    # Function that takes in path to directory of images to upload to database and returns a success or failure message.
    def bulk_upload(self, image_directory_path, database_path=None):
        try:
            # Get database from config file.
            if database_path is None:
                config_path = get_resource_path("db_config.json")
                with open(config_path, "r") as config_file:
                    config = json.load(config_file)
                database_path = get_resource_path(config["database_path"])

            # Get models from config file.
            config_path = get_resource_path("model_config.json")
            with open(config_path, "r") as config_file:
                config = json.load(config_file)
            model_name = config["model_name"]
            detector_backend = config["detector_backend"]
            face_confidence_threshold = config["face_confidence_threshold"]

            # Call helper functions for each image file.
            total_files_uploaded = 0
            embedding_outputs = []
            for root, dirs, files in os.walk(image_directory_path):
                for filename in files:
                    image_path = os.path.join(root, filename)
                    if filename.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".gif", ".bmp")
                    ):
                        status, value = detect_faces_and_get_embeddings(
                            image_path,
                            model_name,
                            detector_backend,
                            face_confidence_threshold,
                        )
                        if status:
                            total_files_uploaded += 1
                            embedding_outputs.extend(value)

                    if total_files_uploaded % 100 == 0:
                        log_info(
                            "Successfully converted file "
                            + str(total_files_uploaded)
                            + " to "
                            "embeddings"
                        )
                    if total_files_uploaded % 1000 == 0:
                        upload_embedding_to_database(embedding_outputs, database_path)
                        embedding_outputs = []
                        log_info(
                            "Successfully uploaded "
                            + str(total_files_uploaded)
                            + " files to database"
                        )

            if len(embedding_outputs) != 0:
                upload_embedding_to_database(embedding_outputs, database_path)
                log_info(
                    "Successfully uploaded "
                    + str(total_files_uploaded)
                    + " files to database"
                )

            return (
                "Successfully uploaded "
                + str(total_files_uploaded)
                + " files to database"
            )
        except Exception as e:
            return f"An error occurred: {str(e)}"

    # Function that takes in path to image and returns all images that have the same person.
    def find_face(self, image_file_path, database_path=None, toggle_faiss=True):
        try:
            # Get database from config file.
            if database_path is None:
                config_path = get_resource_path("db_config.json")
                with open(config_path, "r") as config_file:
                    config = json.load(config_file)
                database_path = get_resource_path(config["database_path"])

            # Get models from config file.
            config_path = get_resource_path("model_config.json")
            with open(config_path, "r") as config_file:
                config = json.load(config_file)
            model_name = config["model_name"]
            detector_backend = config["detector_backend"]
            threshold = config["cosine-threshold"]
            face_confidence_threshold = config["face_confidence_threshold"]

            # Call helper functions to get similar images.
            filename = os.path.basename(image_file_path)
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                status, embedding_outputs = detect_faces_and_get_embeddings(
                    image_file_path,
                    model_name,
                    detector_backend,
                    face_confidence_threshold,
                )
                matching_image_paths = []
                if status:
                    for embedding_output in embedding_outputs:
                        if toggle_faiss:
                            output = cosine_similarity_search_faiss(
                                embedding_output["embedding"],
                                database_path,
                                threshold=threshold,
                            )
                        else:
                            output = cosine_similarity_search(
                                embedding_output["embedding"],
                                database_path,
                                threshold=threshold,
                            )
                        matching_image_paths.extend(output)
                    return matching_image_paths
                else:
                    return "Error: Provided image does not have any face"
            else:
                return "Error: Provided file is not of image type"
        except Exception as e:
            return f"An error occurred: {str(e)}"

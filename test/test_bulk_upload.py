import unittest
import chromadb
import json

from src.facematch.interface import FaceMatchModel
from src.facematch.utils.resource_path import (get_config_path, get_resource_path)


class TestBulkUpload(unittest.TestCase):

    def setUp(self):
        self.image_directory_path = get_resource_path("sample_db")
        self.collection_name = "sample"
        # get model name in order to determine full collection name
        config_path = get_config_path("model_config.json")
        self.client = chromadb.HttpClient(host='localhost', port=8000)
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
        self.model_name = config["model_name"]

    def test_bulk_upload_success(self):
        face_match_object = FaceMatchModel()
        result = face_match_object.bulk_upload(
            self.image_directory_path, self.collection_name
        )
        self.assertEqual("12", result.split(" ")[2])

    def tearDown(self):
        try:
            self.client.delete_collection(f"{self.collection_name}_{self.model_name.lower()}")
        except Exception:
            pass

if __name__ == "__main__":
    unittest.main()

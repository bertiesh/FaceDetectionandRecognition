import os
import unittest

from src.facematch.interface import FaceMatchModel


class TestBulkUpload(unittest.TestCase):

    def setUp(self):
        current_dir = os.path.dirname(__file__)
        self.image_directory_path = os.path.join(
            current_dir, "..", "resources", "sample_images"
        )
        self.database_path = os.path.join(current_dir, "..", "resources", "test_db.csv")

    def test_bulk_upload_success(self):
        face_match_object = FaceMatchModel()
        result = face_match_object.bulk_upload(
            self.image_directory_path, self.database_path
        )
        self.assertEqual("Successfully uploaded to database", result)

    def tearDown(self):
        if os.path.exists(self.database_path):
            os.remove(self.database_path)


if __name__ == "__main__":
    unittest.main()

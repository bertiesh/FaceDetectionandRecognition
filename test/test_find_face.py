import os
import unittest

from src.facematch.interface import FaceMatchModel


class TestMatchFace(unittest.TestCase):

    def setUp(self):
        current_dir = os.path.dirname(__file__)
        self.image_directory_path = os.path.join(
            current_dir, "..", "resources", "sample_images"
        )
        self.database_path = os.path.join(current_dir, "..", "resources", "test_db.csv")
        self.image_file_path = os.path.join(
            current_dir, "..", "resources", "test_image.jpg"
        )

    def test_match_face_success(self):
        face_match_object = FaceMatchModel()
        face_match_object.bulk_upload(self.image_directory_path, self.database_path)
        matching_images = face_match_object.find_face(
            self.image_file_path, self.database_path
        )
        self.assertEqual(1, len(matching_images))
        self.assertEqual("me.png", os.path.basename(matching_images[0]))

    def tearDown(self):
        if os.path.exists(self.database_path):
            os.remove(self.database_path)


if __name__ == "__main__":
    unittest.main()

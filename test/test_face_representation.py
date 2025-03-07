import unittest
import numpy as np

from src.facematch.face_representation import detect_faces_and_get_embeddings
from src.facematch.utils.resource_path import get_resource_path


class TestApp(unittest.TestCase):

    def test_face_representation(self):
        image_path = get_resource_path("sample_images/single.jpg")
        status, result = detect_faces_and_get_embeddings(
            image_path, "ArcFace", "yolov8"
        )
        assert status is True
        assert len(result[0]["embedding"]) == 512
        assert np.all(np.abs(np.array(result[0]["bbox"]) - [226, 201, 83, 119])/[226, 201, 83, 119] < 10)

if __name__ == "__main__":
    unittest.main()

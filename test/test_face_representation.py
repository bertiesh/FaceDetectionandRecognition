import unittest

from src import face_representation


class TestApp(unittest.TestCase):

    def test_face_representation(self):
        result = face_representation.detect_faces_and_get_embeddings(
            "../resources/sample_images/single.jpg"
        )
        assert len(result[0]["embedding"]) == 512
        assert result[0]["region"] == (222, 200, 90, 120)


if __name__ == "__main__":
    unittest.main()

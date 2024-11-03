from typing import TypedDict

from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import (BatchDirectoryInput,
                                             BatchFileInput, BatchFileResponse,
                                             FileResponse, ResponseBody,
                                             TextResponse)

from src.facematch.interface import FaceMatchModel
from src.facematch.logger import log_info

server = MLServer(__name__)

# create an instance of the model
face_match_model = FaceMatchModel()


class FindFaceInputs(TypedDict):
    image_paths: BatchFileInput


class FindFaceParameters(TypedDict): ...


class BulkUploadInputs(TypedDict):
    directory_paths: BatchDirectoryInput


class BulkUploadParameters(TypedDict): ...


@server.route("/findface")
def find_face_endpoint(
    inputs: FindFaceInputs, parameters: FindFaceParameters
) -> ResponseBody:
    input_file_paths = [item.path for item in inputs["image_paths"].files]
    results = face_match_model.find_face(input_file_paths[0])
    log_info(results)
    image_results = [FileResponse(file_type="img", path=res) for res in results]

    return ResponseBody(root=BatchFileResponse(files=image_results))


@server.route("/bulkupload")
def bulk_upload_endpoint(
    inputs: BulkUploadInputs, parameters: BulkUploadParameters
) -> ResponseBody:
    input_directory_paths = [
        item.path for item in inputs["directory_paths"].directories
    ]
    log_info(input_directory_paths[0])
    response = face_match_model.bulk_upload(input_directory_paths[0])

    return ResponseBody(root=TextResponse(value=response))


server.run()

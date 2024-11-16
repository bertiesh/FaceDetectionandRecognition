from typing import List, TypedDict

from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (BatchDirectoryInput,
                                             BatchFileInput, BatchFileResponse,
                                             FileResponse, ResponseBody,
                                             TextResponse, EnumVal, TaskSchema, InputSchema, InputType, ParameterSchema, EnumParameterDescriptor, TextParameterDescriptor)

from src.facematch.interface import FaceMatchModel
from src.facematch.logger import log_info

server = MLServer(__name__)

server.add_app_metadata(
    name="Face Recognition and Matching",
    author="FaceMatch Team",
    version="0.1.0",
    info=load_file_as_string("src/app-info.md"),
)


available_databases: List[str] = []

   
def get_ingest_query_image_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[
            InputSchema(
                key="image_paths",
                label="Image Paths",
                input_type=InputType.BATCHFILE,
            )
        ],
        parameters=[
        ],
    )


# create an instance of the model
face_match_model = FaceMatchModel()

class FindFaceInputs(TypedDict):
    image_paths: BatchFileInput


class FindFaceParameters(TypedDict): ...

@server.route("/findface", order=1, short_title="Find Matching Faces",task_schema_func=get_ingest_query_image_task_schema)
def find_face_endpoint(
    inputs: FindFaceInputs, parameters: FindFaceParameters
) -> ResponseBody:
    input_file_paths = [item.path for item in inputs["image_paths"].files]
    results = face_match_model.find_face(input_file_paths[0])
    log_info(results)
    image_results = [FileResponse(file_type="img", path=res, title=res) for res in results]

    return ResponseBody(root=BatchFileResponse(files=image_results))


def get_ingest_images_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[
            InputSchema(
                key="directory_paths",
                label="Image Directories",
                input_type=InputType.BATCHDIRECTORY,
            )
        ],
        parameters=[
        ],
    )

class BulkUploadInputs(TypedDict):
    directory_paths: BatchDirectoryInput


class BulkUploadParameters(TypedDict): ...


@server.route("/bulkupload", order=0, short_title="Upload Images to Database",task_schema_func=get_ingest_images_task_schema)
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

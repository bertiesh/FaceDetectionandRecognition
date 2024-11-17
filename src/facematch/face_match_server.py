import os
from typing import List, TypedDict

from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (BatchDirectoryInput,
                                             BatchFileInput, BatchFileResponse,
                                             EnumParameterDescriptor, EnumVal,
                                             FileResponse, InputSchema,
                                             InputType, ParameterSchema,
                                             ResponseBody, TaskSchema,
                                             TextParameterDescriptor,
                                             TextResponse)

from src.facematch.interface import FaceMatchModel
from src.facematch.logger import log_info

server = MLServer(__name__)

# Add static location for app-info.md file
script_dir = os.path.dirname(os.path.abspath(__file__))
info_file_path = os.path.join(script_dir, "..", "app-info.md")

server.add_app_metadata(
    name="Face Recognition and Matching",
    author="FaceMatch Team",
    version="0.1.0",
    info=load_file_as_string(info_file_path),
)


available_databases: List[str] = ["Create a new database"]


def get_ingest_query_image_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[
            InputSchema(
                key="image_paths",
                label="Image Path",
                input_type=InputType.BATCHFILE,
            )
        ],
        parameters=[
            ParameterSchema(
                key="database_path",
                label="Database Path",
                value=EnumParameterDescriptor(
                    enum_vals=[
                        EnumVal(key=database_name, label=database_name)
                        for database_name in available_databases[1:]
                    ],
                    message_when_empty="No databases found",
                    default=(
                        available_databases[0] if len(available_databases) > 0 else ""
                    ),
                ),
            ),
        ],
    )


# create an instance of the model
face_match_model = FaceMatchModel()


class FindFaceInputs(TypedDict):
    image_paths: BatchFileInput


class FindFaceParameters(TypedDict):
    database_path: str


@server.route(
    "/findface",
    order=1,
    short_title="Find Matching Faces",
    task_schema_func=get_ingest_query_image_task_schema,
)
def find_face_endpoint(
    inputs: FindFaceInputs, parameters: FindFaceParameters
) -> ResponseBody:
    input_file_paths = [item.path for item in inputs["image_paths"].files]
    results = face_match_model.find_face(
        input_file_paths[0], parameters["database_path"]
    )
    log_info(results)
    image_results = [
        FileResponse(file_type="img", path=res, title=res) for res in results
    ]

    return ResponseBody(root=BatchFileResponse(files=image_results))


def get_ingest_images_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[
            InputSchema(
                key="directory_paths",
                label="Image Directory",
                input_type=InputType.BATCHDIRECTORY,
            )
        ],
        parameters=[
            ParameterSchema(
                key="database_path",
                label="Database Path",
                value=TextParameterDescriptor(default="data/database.csv"),
            ),
            ParameterSchema(
                key="dropdown_database_path",
                label="Database Path",
                value=EnumParameterDescriptor(
                    enum_vals=[
                        EnumVal(key=database_name, label=database_name)
                        for database_name in available_databases
                    ],
                    message_when_empty="No databases found",
                    default=(
                        available_databases[0] if len(available_databases) > 0 else ""
                    ),
                ),
            ),
        ],
    )


class BulkUploadInputs(TypedDict):
    directory_paths: BatchDirectoryInput


class BulkUploadParameters(TypedDict):
    database_path: str
    dropdown_database_path: str


@server.route(
    "/bulkupload",
    order=0,
    short_title="Upload Images to Database",
    task_schema_func=get_ingest_images_task_schema,
)
def bulk_upload_endpoint(
    inputs: BulkUploadInputs, parameters: BulkUploadParameters
) -> ResponseBody:
    if parameters["dropdown_database_path"] == "Create a new database":
        available_databases.append(parameters["database_path"])
    else:
        parameters["database_path"] = parameters["dropdown_database_path"]
    input_directory_paths = [
        item.path for item in inputs["directory_paths"].directories
    ]
    log_info(input_directory_paths[0])
    response = face_match_model.bulk_upload(
        input_directory_paths[0], parameters["database_path"]
    )

    return ResponseBody(root=TextResponse(value=response))


server.run()

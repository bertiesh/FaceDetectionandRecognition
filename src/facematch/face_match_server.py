import os
from pathlib import Path
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
from src.facematch.utils.logger import log_info
from src.facematch.utils.resource_path import get_resource_path

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

# Initialize with "Create a new database" value used in frontend to take new file name entered by user
available_databases: List[str] = ["Create a new database"]

# Load all available datasets under resources/data folder
database_directory_path = get_resource_path("data")
csv_files = list({file.stem for file in Path(database_directory_path).glob("*.csv")})

available_databases.extend(csv_files)


# Frontend Task Schema defining inputs and paraneters that users can enter
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
                key="database_name",
                label="Database Name",
                value=EnumParameterDescriptor(
                    enum_vals=[
                        EnumVal(key=database_name, label=database_name)
                        for database_name in available_databases[1:]
                    ],
                    message_when_empty="No databases found",
                    default=(available_databases[0]),
                ),
            ),
        ],
    )


# create an instance of the model
face_match_model = FaceMatchModel()


# Inputs and parameters for the findface endpoint
class FindFaceInputs(TypedDict):
    image_paths: BatchFileInput


class FindFaceParameters(TypedDict):
    database_name: str


# Endpoint that is used to find matches to a query image
@server.route(
    "/findface",
    order=1,
    short_title="Find Matching Faces",
    task_schema_func=get_ingest_query_image_task_schema,
)
def find_face_endpoint(
    inputs: FindFaceInputs, parameters: FindFaceParameters
) -> ResponseBody:

    # Get list of file paths from input
    input_file_paths = [item.path for item in inputs["image_paths"].files]

    # Convert database name to relative path to data directory in resources folder
    parameters["database_name"] = os.path.join(
        "data", parameters["database_name"] + ".csv"
    )

    # Call model function to find matches
    results = face_match_model.find_face(
        input_file_paths[0], parameters["database_name"]
    )
    log_info(results)

    # Create response object
    image_results = [
        FileResponse(file_type="img", path=res, title=res) for res in results
    ]

    return ResponseBody(root=BatchFileResponse(files=image_results))


# Frontend Task Schema defining inputs and paraneters that users can enter
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
                key="database_name",
                label="Database Name",
                value=TextParameterDescriptor(default="SampleDatabase"),
            ),
            ParameterSchema(
                key="dropdown_database_name",
                label="Database Name",
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


# Inputs and parameters for the bulkupload endpoint
class BulkUploadInputs(TypedDict):
    directory_paths: BatchDirectoryInput


class BulkUploadParameters(TypedDict):
    database_name: str
    dropdown_database_name: str


# Endpoint to allow users to upload images to database
@server.route(
    "/bulkupload",
    order=0,
    short_title="Upload Images to Database",
    task_schema_func=get_ingest_images_task_schema,
)
def bulk_upload_endpoint(
    inputs: BulkUploadInputs, parameters: BulkUploadParameters
) -> ResponseBody:
    # If dropdown value chosen is Create a new database, then add database path to available databases, otherwise set database path to dropdown value
    if parameters["dropdown_database_name"] == "Create a new database":
        available_databases.append(parameters["database_name"])
    else:
        parameters["database_name"] = parameters["dropdown_database_name"]

    # Convert database name to absolute path to database in resources directory
    parameters["database_name"] = os.path.join(
        "data", parameters["database_name"] + ".csv"
    )
    # Get list of directory paths from input
    input_directory_paths = [
        item.path for item in inputs["directory_paths"].directories
    ]
    log_info(input_directory_paths[0])
    # Call the model function
    response = face_match_model.bulk_upload(
        input_directory_paths[0], parameters["database_name"]
    )

    return ResponseBody(root=TextResponse(value=response))


server.run()

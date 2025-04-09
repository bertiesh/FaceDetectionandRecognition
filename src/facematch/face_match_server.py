import json
import os
import chromadb
from typing import List, TypedDict

from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (TextInput, BatchTextResponse,
                                             BatchDirectoryInput, BatchFileInput,
                                             DirectoryInput, BatchFileResponse,
                                             EnumParameterDescriptor, EnumVal,
                                             FileResponse,
                                             FloatRangeDescriptor, InputSchema,
                                             InputType, ParameterSchema,
                                             RangedFloatParameterDescriptor,
                                             ResponseBody, TaskSchema,
                                             TextParameterDescriptor,
                                             TextResponse)

from src.facematch.interface import FaceMatchModel
from src.facematch.utils.GPU import check_cuDNN_version
from src.facematch.utils.logger import log_info

DBclient = chromadb.HttpClient(host='localhost', port=8000)

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

# Initialize with "Create a new collection" value used in frontend to take new file name entered by user
available_collections: List[str] = ["Create a new collection"]

# Load all available collections from chromaDB
existing_collections = [collection.split('_')[0] for collection in DBclient.list_collections()]
available_collections.extend(existing_collections)

# Read default similarity threshold from config file
config_path = os.path.join(script_dir, "config", "model_config.json")
with open(config_path, "r") as config_file:
    config = json.load(config_file)

default_threshold = config["cosine-threshold"]


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
                key="collection_name",
                label="Collection Name",
                value=EnumParameterDescriptor(
                    enum_vals=[
                        EnumVal(key=collection_name, label=collection_name)
                        for collection_name in available_collections[1:]
                    ],
                    message_when_empty="No collections found",
                    default=(available_collections[0]),
                ),
            ),
            ParameterSchema(
                key="similarity_threshold",
                label="Similarity Threshold",
                value=RangedFloatParameterDescriptor(
                    range=FloatRangeDescriptor(min=-1.0, max=1.0),
                    default=default_threshold,
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
    collection_name: str
    similarity_threshold: float


# Endpoint that is used to find matches to a query image
@server.route(
    "/findface",
    order=1,
    short_title="Find Matching Faces For Single Image",
    task_schema_func=get_ingest_query_image_task_schema,
)
def find_face_endpoint(
    inputs: FindFaceInputs, parameters: FindFaceParameters
) -> ResponseBody:

    # Get list of file paths from input
    input_file_paths = [item.path for item in inputs["image_paths"].files]

    # Check CUDNN compatability
    check_cuDNN_version()

    # Call model function to find matches
    status, results = face_match_model.find_face(
        input_file_paths[0],
        parameters["similarity_threshold"],
        parameters["collection_name"],
    )
    log_info(status)
    log_info(results)

    # Create response object of images if status is True
    if not status:
        return ResponseBody(root=TextResponse(value=results))

    image_results = [
        FileResponse(file_type="img", path=res, title=res) for res in results
    ]

    return ResponseBody(root=BatchFileResponse(files=image_results))


# Frontend Task Schema defining inputs and paraneters that users can enter
def get_ingest_bulk_query_image_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[
            InputSchema(
                key="query_directory",
                label="Query Directory",
                input_type=InputType.DIRECTORY,
            )
        ],
        parameters=[
            ParameterSchema(
                key="collection_name",
                label="Collection Name",
                value=EnumParameterDescriptor(
                    enum_vals=[
                        EnumVal(key=collection_name, label=collection_name)
                        for collection_name in available_collections[1:]
                    ],
                    message_when_empty="No collections found",
                    default=(available_collections[0]),
                ),
            ),
            ParameterSchema(
                key="similarity_threshold",
                label="Similarity Threshold",
                value=RangedFloatParameterDescriptor(
                    range=FloatRangeDescriptor(min=-1.0, max=1.0),
                    default=default_threshold,
                ),
            ),
        ],
    )


# Inputs and parameters for the findfacebulk endpoint
class FindFaceBulkInputs(TypedDict):
    query_directory: DirectoryInput


class FindFaceBulkParameters(TypedDict):
    collection_name: str
    similarity_threshold: float


# Endpoint that is used to find matches to a set of query images
@server.route(
    "/findfacebulk",
    order=2,
    short_title="Find Matching Faces In Bulk",
    task_schema_func=get_ingest_bulk_query_image_task_schema,
)
def find_face_bulk_endpoint(
    inputs: FindFaceBulkInputs, parameters: FindFaceBulkParameters
) -> ResponseBody:

    # Check CUDNN compatability
    check_cuDNN_version()

    # Call model function to find matches
    status, results = face_match_model.find_face_bulk(
        inputs["query_directory"].path,
        parameters["similarity_threshold"],
        parameters["collection_name"],
    )
    log_info(status)

    return ResponseBody(root=TextResponse(value=str(results)))

# Inputs and parameters for the findfacebulk endpoint
class FindFaceBulkTestingInputs(TypedDict):
    query_directory: DirectoryInput


class FindFaceBulkTestingParameters(TypedDict):
    collection_name: str


# Endpoint that is used to find matches to a set of query images
# Does not filter by the similarity theshold and returns all results with similarity scores, file paths and
# face index within given query (if multiple faces found in the query, are the results for face 0, 1, etc.)
@server.route(
    "/findfacebulktesting",
    order=5,
)
def find_face_bulk_testing_endpoint(
    inputs: FindFaceBulkTestingInputs, parameters: FindFaceBulkTestingParameters
) -> ResponseBody:

    # Check CUDNN compatability
    check_cuDNN_version()

    # Call model function to find matches
    status, results = face_match_model.find_face_bulk(
        inputs["query_directory"].path,
        None,
        parameters["collection_name"],
        similarity_filter=False
    )
    log_info(status)

    return ResponseBody(root=TextResponse(value=str(results)))




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
                key="dropdown_collection_name",
                label="Choose Collection",
                value=EnumParameterDescriptor(
                    enum_vals=[
                        EnumVal(key=collection_name, label=collection_name)
                        for collection_name in available_collections
                    ],
                    message_when_empty="No collections found",
                    default=(
                        available_collections[0] if len(available_collections) > 0 else ""
                    ),
                ),
            ),
            ParameterSchema(
                key="collection_name",
                label="New Collection Name (Optional)",
                value=TextParameterDescriptor(default="sample"),
            ),
        ],
    )


# Inputs and parameters for the bulkupload endpoint
class BulkUploadInputs(TypedDict):
    directory_paths: BatchDirectoryInput


class BulkUploadParameters(TypedDict):
    dropdown_collection_name: str
    collection_name: str


# Endpoint to allow users to upload images to chromaDB
@server.route(
    "/bulkupload",
    order=0,
    short_title="Upload Images to collection in chromaDB database",
    task_schema_func=get_ingest_images_task_schema,
)
def bulk_upload_endpoint(
    inputs: BulkUploadInputs, parameters: BulkUploadParameters
) -> ResponseBody:
    # If dropdown value chosen is Create a new collection, then add collection to available collections, otherwise set
    # collection to dropdown value
    if parameters["dropdown_collection_name"] != "Create a new collection":
        parameters["collection_name"] = parameters["dropdown_collection_name"]

    new_collection_name = parameters["collection_name"]

    # Check CUDNN compatability
    check_cuDNN_version()

    # Get list of directory paths from input
    input_directory_paths = [
        item.path for item in inputs["directory_paths"].directories
    ]
    log_info(input_directory_paths[0])
    # Call the model function
    response = face_match_model.bulk_upload(
        input_directory_paths[0], parameters["collection_name"]
    )

    if response.startswith("Successfully uploaded") and response.split(" ")[2] != "0":
        # Some files were uploaded
        if parameters["dropdown_collection_name"] == "Create a new collection":
            # Add new collection to available collections if collection name is not already in available collections
            if parameters["collection_name"] not in available_collections:
                available_collections.append(new_collection_name)
    return ResponseBody(root=TextResponse(value=response))


# Inputs and parameters for the bulkupload endpoint
class DeleteCollectionInputs(TypedDict):
    collection_name: TextInput
    model_name: TextInput
    detector_backend: TextInput


class DeleteCollectionParameters(TypedDict):
    pass

# Endpoint for deleting collections from ChromaDB
@server.route(
    "/deletecollection",
    order=3,
)
def delete_collection_endpoint(
    inputs: DeleteCollectionInputs, parameters: DeleteCollectionParameters
) -> ResponseBody:
    
    responseValue = ""

    try:
        collection_name = inputs["collection_name"].text
        model_name = inputs["model_name"].text.lower()
        detector_backend = inputs["detector_backend"].text.lower()
        DBclient.delete_collection(f"{collection_name}_{detector_backend}_{model_name}")
        responseValue = f'Successfully deleted {collection_name}_{detector_backend}_{model_name}'
        log_info(responseValue)
    except Exception:
        responseValue = f'Collection {collection_name}_{detector_backend}_{model_name} does not exist.'
        log_info(responseValue)   

    return ResponseBody(root=TextResponse(value=responseValue))


# Inputs and parameters for the bulkupload endpoint
class ListCollectionsInputs(TypedDict):
    pass


class ListCollectionsParameters(TypedDict):
    pass

# Endpoint for listing all ChromaDB collections
@server.route(
    "/listcollections",
    order=4,
)
def list_collections_endpoint(
    inputs: ListCollectionsInputs, parameters: ListCollectionsParameters
) -> ResponseBody:
    
    responseValue = None

    try:
        responseValue = DBclient.list_collections()
        log_info(responseValue)
    except Exception:
        responseValue = ['Failed to List Collections']
        log_info(responseValue)        

    return ResponseBody(root=BatchTextResponse(texts=[TextResponse(value=collection) for collection in responseValue]))




server.run()

# FACEMATCH

FaceMatch is a system for identifying facial matches within an image database. With FaceMatch, users can create a database of images of people and, by uploading a new image, quickly find any matches for the person of interest within the database. 

Built with a client-server architecture using Flask-ML, FaceMatch provides structured support for efficient client-server communication tailored to ML applications.


# Getting started

## Prerequisites

- Python 3.12 
- Virtual environment support (recommended but optional)

## Installation

### Clone the repository

```
git clone https://github.com/RigvedManoj/FaceMatch.git
cd FaceMatch
```

### SetUp Virtual environment

```
python -m venv facematch-env
source facematch-env/bin/activate  

# On Windows: facematch-env\Scripts\activate
```

#### With Conda

```
conda create -n facematch-env python=3.12
conda activate facematch-env
```

### Install dependencies

```
pip install -r requirements.txt
```
# Usage

## CLI

### Start the server
```
python -m src.facematch.face_match_server
```

### Task 1: Upload images to database
```
python -m src.Sample_Client.sample_bulk_upload_client <path_to_directory_of_images>

Run with Sample images:
python -m src.Sample_Client.sample_bulk_upload_client resources/sample_images
```

### Task 2: Find matching faces
```
python -m src.Sample_Client.sample_find_face_client <path_to_image>

Run with Sample image:
python -m src.Sample_Client.sample_find_face_client resources/test_image.jpg
```

## Rescue-Box frontend

### Start the server
```
python -m src.facematch.face_match_server
```

### Use Rescue-Box-Desktop

- Install Rescue-Box from [link](https://github.com/UMass-Rescue/RescueBox-Desktop)

- Open Rescue-Box-Desktop and resgiter the model by adding the server IP address and port number in which the server is running.

- Choose the model from list of available models under the **MODELS** tab.

- Checkout the Inspect page to learn more about using the model.

# For Developers
## Run unit tests

```
python -m unittest discover test
```
- #### Run individual test

```
python -m unittest test.<test_file_name>
```



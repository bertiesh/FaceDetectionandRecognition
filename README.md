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

### Run unit tests

```
python -m unittest discover test
```
- #### Run individual test

```
python -m unittest test.<test_file_name>
```



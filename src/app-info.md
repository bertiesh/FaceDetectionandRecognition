# FaceMatch

FaceMatch is an advanced system designed for identifying facial matches within an image database. This platform enables users to create comprehensive image databases and facilitates efficient searches to find matches for individuals by uploading query images.

## 1. Creating a Database of Individuals

To populate a database with images, utilize the **Upload Images to Database** endpoint.

### Inputs

- **Image Directory:** A directory containing images to be added to the database.

    - For multiple directories, repeat the upload process sequentially, ensuring that each previous directory has been successfully added before proceeding.

- **Choose database:** A dropdown menu to select either an existing database to upload the images to or the option to create a new database.

- **New Database Name (Optional):** A text input field to enter the name for the new database. 

> NOTE: 
> Use only if you would like to create a new database.
> Ensure that the "Create a new database" option is selected in 'Choose database' dropdown.

### Outputs

- Refer to the results section for the status of the upload process.

## 2. Search for matches 

- To search for facial matches within the existing database, use the **Find Matching Faces** endpoint.

### Inputs

- **Image Path:** A query image to be compared against the database.

- **Database Name:** Choose the database to search within.

- **Similarity Threshold:** A threshold value to determine the minimum similarity score required for a match to be considered a positive match. 

Default value provides a tradeoff between two things, it tries to ensure we find the right person when they are in the database while also avoiding saying we found someone when they aren't there.  This setting can be adjusted depending on whether you want to focus more on finding as many matches as possible (decrease threshold) or being extra careful to avoid wrong ones (increase threshold).


**Guide to tuning threshold:** 
- Increase the threshold value to narrow down the search to higher similarity faces. 

   **Sample case:** You could increase threshold if the search image is clear and up-to-date, so you can look for a match with a high degree of confidence.

- Decrese the threshold if you wish to broaden the search to include more variability in faces. 

    **Sample case:** You could decrease threshold if the search image is blurry or outdated, so you are open to some variability in the possible matches.

### Outputs 

- Refer to the results section for the matches found within the database.
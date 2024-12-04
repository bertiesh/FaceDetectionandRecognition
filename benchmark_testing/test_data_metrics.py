import re

import pandas as pd
from sklearn.metrics import accuracy_score

# Load the CSV file
file_path = "path\\to\\output\\csv"  # Path to csv file containing top-n matches

# Sample csv path
# file_path = "<path to dataset folder>\\LFWdataset\\output.csv"

data = pd.read_csv(file_path)

# Extract ground truth names (base names without numeric suffixes)
data["ground_truth"] = data["filename"].apply(
    lambda x: re.match(r"(.+?)_\d+\.jpg", x).group(1)
)


# Define a function to check if ground truth matches any predicted name
def check_match(row):
    # Check for NaN in 'result' column
    if pd.isna(row["result"]):
        return False  # No match if 'result' is NaN
    # Split the result by spaces to get individual file names
    predicted_names = row["result"].split()
    # Extract base names from each filename
    predicted_base_names = [
        re.match(r"(.+?)_\d+\.jpg", name).group(1) for name in predicted_names
    ]
    # Check if ground_truth matches any of the base names
    return row["ground_truth"] in predicted_base_names


# Apply the function to each row to create a 'predicted' column with boolean values
data["predicted"] = data.apply(check_match, axis=1)

# Ground truth column set to True for all entries, as each `filename` is assumed to be a positive example
data["true_label"] = True

# Calculate metrics
accuracy = accuracy_score(data["true_label"], data["predicted"])

# Output the results
print("Accuracy:", accuracy)

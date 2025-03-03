import re
import os
import csv
from dotenv import load_dotenv

import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)



def calculate_tpr_fpr(true_labels, predicted_labels):
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Extract values from confusion matrix
    TP = cm[1, 1]  # True Positive
    TN = cm[0, 0]  # True Negative
    FP = cm[0, 1]  # False Positive
    FN = cm[1, 0]  # False Negative

    # Calculate TPR (True Positive Rate) and FPR (False Positive Rate)
    tpr = TP / (TP + FN)  # TPR = TP / (TP + FN)
    fpr = FP / (FP + TN)  # FPR = FP / (FP + TN)

    #calculate TNR and FNR
    tnr = TN / (TN + FP)
    fnr = FN / (FN + TP)
    return tpr, fpr, tnr, fnr

# Define a function to check if ground truth matches any predicted name
def check_match(row, n, true_label_index=None):
    # Check for NaN in 'result' column
    if pd.isna(row["result"]):
        return False  # No match if 'result' is NaN
    # Split the result by spaces to get individual file names
    predicted_names = row["result"].split()[0:n]
    
    # Extract base names from each filename
    
    predicted_base_names = []
    for name in predicted_names:
        base_filename = os.path.basename(name.strip())
        match = re.match(r"(.+?)_\d+\.jpg", base_filename)
        if match:
            predicted_base_names.append(match.group(1))

    # Extract base name from the input image
    ground_truth = re.match(r"(.+?)_\d+\.jpg", row["filename"]).group(1)

    if not row["true_label"]:
        return len(predicted_base_names) > 0

    # Check if the ground truth matches any of the filtered base names
    return ground_truth in predicted_base_names

# Extract ground truth names (base names without numeric suffixes)
def extract_ground_truth(x):
    match = re.match(r"(.+?)_\d+\.jpg", x)
    return match.group(1) if match else None

load_dotenv()

top_n = ['top_1', 'top_5', 'top_10']
N = [1, 5, 10]

for top_n, n in zip(top_n, N):
    # csv output files directory
    output_directory = os.getenv("OUTPUT_CSV_DIRECTORY")

    # benchmark results file path
    benchmark_results_path = os.getenv("BENCHMARK_RESULTS_PATH")+f'_{top_n}.csv'

    # Sample csv path
    # file_path = "<path to dataset folder>\\LFWdataset\\output.csv"

    with open(benchmark_results_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['model', 'similarity threshold', 'accuracy', 'precision', 'recall', 'f1', 'tpr', 'fpr', 'tnr', 'fnr'])

    for filename in os.listdir(output_directory):
        if not filename.endswith('.csv'):
            continue

        model_name, similarity_threshold = os.path.splitext(filename)[0].split("_")[-2:]

        data = pd.read_csv(os.path.join(output_directory, filename))
    
        # Determine the midpoint of the DataFrame
        midpoint = len(data) // 2

        # Set `true_label` column
        data["true_label"] = [True]*400 + [False]*100

        # Apply the function to each row to create a 'predicted' column with boolean values
        data["predicted"] = data.apply(lambda row: check_match(row, n), axis=1)
        tpr, fpr, tnr, fnr = calculate_tpr_fpr(data["true_label"], data["predicted"])

        # Calculate metrics
        accuracy = accuracy_score(data["true_label"], data["predicted"])
        precision = precision_score(data["true_label"], data["predicted"])
        recall = recall_score(data["true_label"], data["predicted"])
        f1 = f1_score(data["true_label"], data["predicted"])

        results = [model_name, similarity_threshold, accuracy, precision, recall, f1, tpr, fpr, tnr, fnr]

        with open(benchmark_results_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(results)


        # Output the results
        print("Model: ", model_name, 'Similarity Threshold: ', similarity_threshold)
        print('tpr: ', tpr, 'fpr: ', fpr, 'tnr: ', tnr, 'fnr: ', fnr)
        print("Acc:", accuracy, "Prec:", precision, "Rec:", recall, "f1: ", f1)
        print('\n\n')

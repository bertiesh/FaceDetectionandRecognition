#!/bin/bash

# FaceMatch End-to-End Test Script
# This script performs a complete test cycle:
# 1. Cleans up existing collections in ChromaDB
# 2. Uploads images from a specified directory
# 3. Performs bulk face matching
# 4. Displays testing metrics

# Default values
DB_NAME="test"
UPLOAD_DIR="./resources/LFWdataset/new_sample_database"
QUERY_DIR="./resources/LFWdataset/sample_queries"
SIMILARITY_THRESHOLD=0.58
TOP_N=5
MODEL_NAME="ArcFace" # Default model name
DETECTOR_NAME="yolov8" # Default detector name
OUTPUT_DIR="./test_results"
USE_CUSTOM_COLLECTION_NAME=false
CLEAN_DB=true

# Display usage information
function show_usage {
    echo "Usage: $0 [OPTIONS]"
    echo "OPTIONS:"
    echo "  -d, --db-name NAME         Database name (default: test)"
    echo "  -u, --upload-dir PATH      Upload directory path (default: ./resources/sample_db)"
    echo "  -q, --query-dir PATH       Query directory path (default: ./resources/sample_queries)"
    echo "  -t, --threshold VALUE      Similarity threshold (default: 0.5)"
    echo "  -o, --output-dir PATH      Output directory for results (default: ./test_results)"
    echo "  -k, --keep-db              Don't clean up existing collections"
    echo "  -c, --custom-db            Use detector and model in collection name"
    echo "  -n, --top-n VALUE          Number of top matches to consider (default: 5)"
    echo "  -h, --help                 Show this help message"
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--db-name) DB_NAME="$2"; shift ;;
        -u|--upload-dir) UPLOAD_DIR="$2"; shift ;;
        -q|--query-dir) QUERY_DIR="$2"; shift ;;
        -t|--threshold) SIMILARITY_THRESHOLD="$2"; shift ;;
        -o|--output-dir) OUTPUT_DIR="$2"; shift ;;
        -k|--keep-db) CLEAN_DB=false ;;
        -c|--custom-db) USE_CUSTOM_COLLECTION_NAME=true ;;
        -n|--top-n) TOP_N="$2"; shift ;;
        -h|--help) show_usage; exit 0 ;;
        *) echo "Unknown parameter: $1"; show_usage; exit 1 ;;
    esac
    shift
done

# Validate required directories
if [ ! -d "$UPLOAD_DIR" ]; then
    echo "Error: Upload directory not found: $UPLOAD_DIR"
    exit 1
fi

if [ ! -d "$QUERY_DIR" ]; then
    echo "Error: Query directory not found: $QUERY_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set Python path to include the project root
export PYTHONPATH=$(pwd)

# Read model and detector names from config
MODEL_CONFIG="./src/facematch/config/model_config.json"
if [ -f "$MODEL_CONFIG" ]; then
    # Improved extraction with proper parsing
    MODEL_NAME=$(grep '"model_name"' "$MODEL_CONFIG" | cut -d'"' -f4)
    DETECTOR_NAME=$(grep '"detector_backend"' "$MODEL_CONFIG" | cut -d'"' -f4)
    
    echo "Using model: $MODEL_NAME"
    echo "Using detector: $DETECTOR_NAME"
    
    # Create a custom collection name based on detector and model if requested
    if [ "$USE_CUSTOM_COLLECTION_NAME" = true ]; then
        ORIGINAL_DB_NAME="$DB_NAME"
        DB_NAME="${MODEL_NAME}_${DETECTOR_NAME}_${ORIGINAL_DB_NAME}"
        echo "Using custom collection name: $DB_NAME"
    fi
else
    echo "Warning: Could not find model config file, using default model settings"
fi

# Function to clean up existing ChromaDB collections
function cleanup_db {
    if [ "$CLEAN_DB" = false ]; then
        echo "==== Skipping database cleanup (--keep-db specified) ===="
        return 0
    fi
    
    echo "==== Cleaning up existing ChromaDB collections ===="
    
    # Create a Python script to list and delete collections
    # If CLEAN_DB is false, list collections but don't delete them
    cat > "$OUTPUT_DIR/cleanup_db.py" <<EOF
import chromadb
import sys

client = chromadb.HttpClient(host='localhost', port=8000)
collections = client.list_collections()

print(f"Found {len(collections)} collections")
for collection in collections:
    if '$CLEAN_DB' == 'true':
        print(f"Deleting collection: {collection}")
        client.delete_collection(collection)
    else:
        print(f"Collection (not deleting): {collection}")

if '$CLEAN_DB' == 'true':
    print("Cleanup complete")
else:
    print("Cleanup skipped, keeping existing collections")
EOF

    # Execute the cleanup script
    python "$OUTPUT_DIR/cleanup_db.py"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to clean up ChromaDB collections"
        echo "Is ChromaDB server running? Start it with: chroma run --path ./resources/data"
        exit 1
    fi
}

# Function to start the ChromaDB server
function start_chromadb {
    echo "==== Starting ChromaDB server ===="
    chroma run --path ./resources/data &
    CHROMA_PID=$!
    echo "ChromaDB server started with PID: $CHROMA_PID"
    
    # Wait for server to start
    echo "Waiting for ChromaDB server to start (10 seconds)..."
    sleep 10
}

# Function to stop the ChromaDB server
function stop_chromadb {
    if [ -n "$CHROMA_PID" ]; then
        echo "==== Stopping ChromaDB server (PID: $CHROMA_PID) ===="
        kill $CHROMA_PID
        wait $CHROMA_PID 2>/dev/null
        echo "ChromaDB server stopped"
    fi
}

# Function to start the FaceMatch server
function start_facematch_server {
    echo "==== Starting FaceMatch server ===="
    python -m src.facematch.face_match_server &
    FACEMATCH_PID=$!
    echo "FaceMatch server started with PID: $FACEMATCH_PID"
    
    # Wait for server to start
    echo "Waiting for FaceMatch server to start (10 seconds)..."
    sleep 10
}

# Function to stop the FaceMatch server
function stop_facematch_server {
    if [ -n "$FACEMATCH_PID" ]; then
        echo "==== Stopping FaceMatch server (PID: $FACEMATCH_PID) ===="
        kill $FACEMATCH_PID
        wait $FACEMATCH_PID 2>/dev/null
        echo "FaceMatch server stopped"
    fi
}

# Function to upload images to the database
function upload_images {
    echo "==== Uploading images to database ($DB_NAME) ===="
    echo "Source directory: $UPLOAD_DIR"
    
    # Start timer
    UPLOAD_START_TIME=$(date +%s)
    
    # Call the bulk upload client
    python -m src.Sample_Client.sample_bulk_upload_client --directory_paths "$UPLOAD_DIR" --collection_name "$DB_NAME"
    UPLOAD_STATUS=$?
    
    # End timer
    UPLOAD_END_TIME=$(date +%s)
    UPLOAD_DURATION=$((UPLOAD_END_TIME - UPLOAD_START_TIME))
    
    echo "Upload completed in $UPLOAD_DURATION seconds"
    
    if [ $UPLOAD_STATUS -ne 0 ]; then
        echo "Error: Failed to upload images to database"
        return 1
    fi
    
    return 0
}

# Function to perform bulk face matching
function perform_face_matching {
    echo "==== Performing bulk face matching ===="
    echo "Query directory: $QUERY_DIR"
    echo "Database: $DB_NAME"
    echo "Similarity threshold: $SIMILARITY_THRESHOLD"
    
    # Important fix: Your system adds "_modelname" to collection names
    # To handle this correctly, we'll use the collection name without manipulation
    # and rely on the database_functions.py to handle any necessary modifications
    
    # Start timer
    MATCH_START_TIME=$(date +%s)
    
    # Create output CSV file path
    OUTPUT_CSV="$OUTPUT_DIR/face_match_results.csv"
    echo "filename,result" > "$OUTPUT_CSV"
    
    # Get the list of query images
    QUERY_IMAGES=($(find "$QUERY_DIR" -type f -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | sort))
    TOTAL_IMAGES=${#QUERY_IMAGES[@]}
    
    echo "Found $TOTAL_IMAGES query images"
    
    # Process each image
    for ((i=0; i<TOTAL_IMAGES; i++)); do
        IMAGE_PATH="${QUERY_IMAGES[$i]}"
        FILENAME=$(basename "$IMAGE_PATH")
        
        echo "Processing image $((i+1))/$TOTAL_IMAGES: $FILENAME"
        
        # Call the face match client
        RESULT=$(python -m src.Sample_Client.sample_find_face_client --file_paths "$IMAGE_PATH" --collection_name "$DB_NAME" --similarity_threshold "$SIMILARITY_THRESHOLD" | grep -v "Matches found" | tr '\n' ' ')
        
        # Save result to CSV
        echo "$FILENAME,$RESULT" >> "$OUTPUT_CSV"
    done
    
    # End timer
    MATCH_END_TIME=$(date +%s)
    MATCH_DURATION=$((MATCH_END_TIME - MATCH_START_TIME))
    
    echo "Face matching completed in $MATCH_DURATION seconds"
    echo "Results saved to: $OUTPUT_CSV"
    
    return 0
}

# Function to calculate and display metrics
function calculate_metrics {
    echo "==== Calculating metrics ===="
    
    # Create a Python script to calculate metrics
    cat > "$OUTPUT_DIR/calculate_metrics.py" <<EOF
import re
import os
import pandas as pd
from sklearn.metrics import confusion_matrix

# Load the CSV file
file_path = "$OUTPUT_DIR/face_match_results.csv"
data = pd.read_csv(file_path)

# Extract ground truth names (base names without numeric suffixes)
def extract_ground_truth(x):
    match = re.match(r"(.+?)_\d+\.jpg", x)
    return match.group(1) if match else None

data["ground_truth"] = data["filename"].apply(extract_ground_truth)

# Get top-N value from environment
top_n = int(os.getenv('TOP_N', 5))
print(f"Evaluating top-{top_n} matches")

# Define a function to check if ground truth matches any predicted name, considering only top-N matches
def check_match(row, top_n=5):
    # Check for NaN in 'result' column
    if pd.isna(row["result"]) or row["result"].strip() == "":
        # No matches found
        return False
    
    # Check if the result contains "Collection does not exist"
    if "Collection does not exist" in row["result"]:
        return False
        
    # Split the result by spaces to get individual file names
    predicted_paths = row["result"].split()
    
    # Limit to top-N matches
    predicted_paths = predicted_paths[:top_n]
    
    # Extract base names from each filename
    predicted_base_names = []
    for path in predicted_paths:
        base_filename = os.path.basename(path.strip())
        match = re.match(r"(.+?)_\d+\.jpg", base_filename)
        if match:
            predicted_base_names.append(match.group(1))
    
    # For checking if a match was found, we only need to know if there's at least one prediction
    return len(predicted_base_names) > 0

# Apply the function to each row to create a 'predicted' column with boolean values
data["predicted"] = data.apply(lambda row: check_match(row, top_n), axis=1)

# Determine which images should have matches
# Assumption: First half should have matches, second half should not
midpoint = len(data) // 2
data["true_label"] = False
data.loc[:midpoint-1, "true_label"] = True

# Add a column to check if the prediction is correct based on whether the predicted names match the ground truth
def is_correct_match(row, top_n=5):
    # If no match was expected and none was found, that's correct
    if not row["true_label"] and not row["predicted"]:
        return True
        
    # If no match was expected but one was found, that's incorrect
    if not row["true_label"] and row["predicted"]:
        return False
        
    # If a match was expected but none was found, that's incorrect
    if row["true_label"] and not row["predicted"]:
        return False
        
    # If a match was expected and found, check if it's the correct person
    if row["true_label"] and row["predicted"]:
        # Get the predicted person names (limited to top-N)
        predicted_person_names = []
        paths = row["result"].split()[:top_n]  # Limit to top-N matches
        for path in paths:
            base_filename = os.path.basename(path.strip())
            match = re.match(r"(.+?)_\d+\.jpg", base_filename)
            if match:
                predicted_person_names.append(match.group(1))
                
        # Check if the ground truth name is in the predicted names
        return row["ground_truth"] in predicted_person_names
    
    return False

data["is_correct"] = data.apply(lambda row: is_correct_match(row, top_n), axis=1)

# Calculate basic metrics
accuracy = data["is_correct"].mean()

# Calculate TP, FP, TN, FN properly
true_positives = sum((data["true_label"] == True) & (data["predicted"] == True) & (data["is_correct"] == True))
false_positives = sum((data["true_label"] == False) & (data["predicted"] == True))
true_negatives = sum((data["true_label"] == False) & (data["predicted"] == False))
false_negatives = sum((data["true_label"] == True) & ((data["predicted"] == False) | 
                                                     ((data["predicted"] == True) & (data["is_correct"] == False))))

# Calculate metrics based on the properly counted values
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# For compatibility with the output format
tp, fp, tn, fn = true_positives, false_positives, true_negatives, false_negatives

# Output the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"True Negatives: {tn}")
print(f"False Negatives: {fn}")

# Save detailed results to a file
with open("$OUTPUT_DIR/detailed_results.txt", "w") as f:
    f.write(f"Filename, Expected Match, Found Match (Top-{top_n}), Correct Match?, Matched Names\n")
    for i, row in data.iterrows():
        expected = "Yes" if row["true_label"] else "No"
        found = "Yes" if row["predicted"] else "No"
        correct = "✓" if row["is_correct"] else "✗"
        
        # Get the matched names for display
        matched_names = ""
        if row["predicted"] and not pd.isna(row["result"]) and "Collection does not exist" not in row["result"]:
            names = []
            # Limit to top-N matches
            for path in row["result"].split()[:top_n]:
                base_name = os.path.basename(path.strip())
                names.append(base_name)
            matched_names = ", ".join(names)
        
        f.write(f"{row['filename']}, {expected}, {found}, {correct}, {matched_names}\n")

# Save metrics to a file
with open("$OUTPUT_DIR/metrics.txt", "w") as f:
    f.write(f"Collection Name: {os.getenv('DB_NAME', 'Unknown')}\n")
    f.write(f"Model: {os.getenv('MODEL_NAME', 'Unknown')}\n")
    f.write(f"Detector: {os.getenv('DETECTOR_NAME', 'Unknown')}\n")
    f.write(f"Similarity Threshold: {os.getenv('SIMILARITY_THRESHOLD', 'Unknown')}\n")
    f.write(f"Evaluation: Top-{top_n} matches\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(f"True Positives: {tp} (correctly found matches)\n")
    f.write(f"False Positives: {fp} (incorrectly found matches)\n")
    f.write(f"True Negatives: {tn} (correctly found no matches)\n")
    f.write(f"False Negatives: {fn} (incorrectly found no matches)\n")

print(f"Metrics saved to: $OUTPUT_DIR/metrics.txt")
print(f"Detailed results saved to: $OUTPUT_DIR/detailed_results.txt")
EOF

    # Set environment variables for the Python script
    export MODEL_NAME
    export DETECTOR_NAME
    export SIMILARITY_THRESHOLD
    export TOP_N
    export DB_NAME
    
    # Execute the metrics calculation script
    python "$OUTPUT_DIR/calculate_metrics.py"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to calculate metrics"
        return 1
    fi
    
    # Display the metrics file
    if [ -f "$OUTPUT_DIR/metrics.txt" ]; then
        echo "==== Test Results ===="
        cat "$OUTPUT_DIR/metrics.txt"
    fi
    
    return 0
}

# Main execution flow
echo "==== FaceMatch End-to-End Test Script ===="
echo "Started at: $(date)"
echo "DB Name: $DB_NAME"
echo "Upload Directory: $UPLOAD_DIR"
echo "Query Directory: $QUERY_DIR"
echo "Similarity Threshold: $SIMILARITY_THRESHOLD"
echo "Keep Existing DB: $([[ "$CLEAN_DB" = false ]] && echo "Yes" || echo "No")"
echo "Custom Collection Name: $([[ "$USE_CUSTOM_COLLECTION_NAME" = true ]] && echo "Yes" || echo "No")"
echo "Top-N Evaluation: $TOP_N"
echo "Output Directory: $OUTPUT_DIR"

# Setup trap to ensure servers are stopped on exit
trap "stop_facematch_server; stop_chromadb; echo 'Test script aborted.'; exit 1" SIGINT SIGTERM

# Start ChromaDB server
start_chromadb

# Clean up existing collections
cleanup_db

# Start FaceMatch server
start_facematch_server

# Upload images
upload_images
if [ $? -ne 0 ]; then
    stop_facematch_server
    stop_chromadb
    exit 1
fi

# Perform face matching
perform_face_matching
if [ $? -ne 0 ]; then
    stop_facematch_server
    stop_chromadb
    exit 1
fi

# Calculate metrics
calculate_metrics

# Stop servers
stop_facematch_server
stop_chromadb

echo "==== Test complete ===="
echo "Ended at: $(date)"
echo "All results saved to: $OUTPUT_DIR"
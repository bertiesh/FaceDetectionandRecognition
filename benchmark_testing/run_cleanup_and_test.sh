#!/bin/bash
# Modified main script (run_all.sh)

# Parse command line arguments
database_name="sample_db"
similarity_threshold=0.50  # Default value
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -t|--threshold) similarity_threshold="$2"; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

# Delete output.csv if it exists
if [ -f "../resources/LFWdataset/output.csv" ]; then
  echo "Deleting ../resources/LFWdataset/output.csv"
  rm "../resources/LFWdataset/output.csv"
fi

# # Delete sample_db.csv and sample_db.bin if they exist
# if [ -f "../resources/data/sample_db.csv" ]; then
#   echo "Deleting ../resources/data/sample_db.csv"
#   rm "../resources/data/sample_db.csv"
# fi

# if [ -f "../resources/data/sample_db.bin" ]; then
#   echo "Deleting ../resources/data/sample_db.bin"
#   rm "../resources/data/sample_db.bin"
# fi

# # Run the required scripts without waiting for key presses
# echo "Running bulk upload script..."
# # Use modified run_bulk_upload_auto.sh script (see below)
# ./run_bulk_upload.sh $database_name

echo "Running face find accuracy script..."
# Pass the similarity threshold to the run_face_find_accuracy.sh script
./run_face_find_accuracy.sh --threshold $similarity_threshold

echo "Running data metrics script..."
python test_data_metrics.py

echo "Running confusion matrix script..."
python test_data_metrics_confusion_matrix.py

echo "All tasks completed."
#!/bin/bash

source ../.env
export PYTHONPATH=$(pwd)/..

# if [ $# -ne 1 ]; then
#     echo -e "Expected 1 argument\n"
#     echo -e "Usage: ./benchmark.sh collection_name\n"
#     read -p "Press any key to exit..."
#     exit 1
# fi

# Collection Name
collection_name="benchmark-sample"
# Output directory for results csv
time_csv_path="$TIME_CSV_PATH"
results_csv_path="$RESULTS_CSV_PATH"
database_directory="$DATABASE_DIRECTORY"
queries_directory="$QUERIES_DIRECTORY"

directories=("./output-csv-dump" "./results-csv")

for dir in "${directories[@]}"; do
    echo "Processing: $dir"
    
    # Check 1: Directory exists
    if [ ! -d "$dir" ]; then
        continue
    fi
    
    # Check 2: Directory is not root
    if [ "$dir" == "/" ]; then
        continue
    fi
    
    # Check 3: Directory is writable
    if [ ! -w "$dir" ]; then
        continue
    fi

    if [ -d "$dir" ]; then
        read -p "Clear $dir? [y/N] " response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            rm -rf "$dir"
            echo "Cleared: $dir"
        else
            echo "Skipped: $dir"
        fi
    else
        echo "Not found: $dir"
    fi
done

mkdir -p "./output-csv-dump"
mkdir -p "./results-csv"

# Start ChromaDB server
chroma run --path ../resources/data &
PID1=$!

# Wait for the DB server to start (adjust time as necessary)
sleep 3

# Start the Python server in the background
python ../src/facematch/face_match_server.py &
server_pid=$!
PID2="$server_pid"
echo "Server started with PID $server_pid"

# Wait for the server to start (adjust time as necessary)
sleep 5

# delete collection from server if it exists
python ../src/Sample_Client/sample_delete_collection_client.py --collection_name "$collection_name"

# Start timer
start_time=$(date +%s)

# Call client script to upload images from to database (the code currently only accepts one directory at a time)
python ../src/Sample_Client/sample_bulk_upload_client.py --directory_paths "$database_directory" --collection_name "$collection_name"

end_time=$(date +%s)
total_time=$((end_time - start_time))
# Print total time taken
echo "Bulk Upload Time: $total_time seconds"

# Write bulk upload time to csv
echo "process, time" > "$time_csv_path"
echo "bulk_upload, $total_time" >> "$time_csv_path"

# Run bulk upload benchmarking to obtain matching results
python ./run_face_find_bulk_benchmark.py --query_directory "$queries_directory" --collection_name "$collection_name"


python ./test_data_metrics_benchmark.py


wait -n $PID1 $PID2

# kill DB and server
kill $PID1 $PID2 2>/dev/null

read -p "Press any key to exit..."
#!/bin/bash

source ../.env

if [ $# -ne 4 ]; then
    echo -e "Expected 4 arguments\n"
    echo -e "Usage: ./run_face_find_benchmark.sh collection_name start_image_index num_images model_name\n"
    read -p "Press any key to exit..."
    exit 1
fi

export PYTHONPATH=$(pwd)/..

# Start the Python server in the background
python ../src/facematch/face_match_server.py &
server_pid=$!
echo "Server started with PID $server_pid"

# Wait for the server to start (adjust time as necessary)
sleep 10

# Define the directory containing the files and the output CSV file
input_directory="$SAMPLE_QUERIES_DIRECTORY" # Path to directory of query images defined in .env file
output_csv="$OUTPUT_CSV_PATH" # Path to csv file containing top-n matches defined in .env file
collection_name=$1 # Name of collection to match against (e.g. sample)
s=$2 # start image index
n=$3 # number of images to query
model=$4 # model name

# Sample directory path
# input_directory = "<path to dataset folder>\\LFWdataset\\sample_queries"
# output_csv = "<path to dataset folder>\\LFWdataset\\output"

# Iterate over each file in the directory
files=($(find "$input_directory" -maxdepth 1 -type f | sort))

# Similarity Thresholds to iterate over
similarity_thresholds=(0.4 0.5 0.6) # 0.2 0.45 0.48 0.54 0.63 0.68 0.7 0.74

# create file to store search times
echo "search_time,model,similarity_threshold" > "${output_csv}_search_times_${model}"

for st in "${similarity_thresholds[@]}"; do
    # Start timer
    start_time=$(date +%s)

    # Initialize the CSV file with headers
    echo "filename,result" > "${output_csv}_${model}_${st}.csv"

    for ((i=s;i<s+n&&i<${#files[@]}; i++)); do
        file="${files[i]}"
        # Ensure it's a file (not a directory)
        if [[ -f "$file" ]]; then
            # Get the filename
            filename=$(basename "$file")

            # Call the Python script and capture its output
            result=$(python ../src/Sample_Client/sample_find_face_client.py --file_paths "$file" --similarity_threshold $st --collection_name "$collection_name" | grep -v "Matches found" | tr '\n' ' ')

            # Append the filename and result as a new row in the CSV file
            echo "$filename,$result" >> "${output_csv}_${model}_${st}.csv"
        fi
    done

    # Calculate total time taken
    end_time=$(date +%s)
    total_time=$((end_time - start_time))

    # Print total time taken
    echo "Total time for threshold = $st taken: $total_time seconds"
    # append search time to output file
    echo "${total_time},${model},${st}" >> "${output_csv}_search_times_${model}"
done

# Stop the server
kill $server_pid
echo "Server stopped"

read -p "Press any key to exit..."
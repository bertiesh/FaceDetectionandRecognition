#!/bin/bash

if [ $# -ne 1 ]; then
    echo -e "Expected 1 argument\n"
    echo -e "Usage: ./run_bulk_upload.sh database_name\n"
    read -p "Press any key to exit..."
    exit 1
fi

source ../.env

export PYTHONPATH=$(pwd)/..

# Start the Python server in the background
python ../src/facematch/face_match_server.py &
server_pid=$!
echo "Server started with PID $server_pid"

# Wait for the server to start (adjust time as necessary)
sleep 10

# Start timer
start_time=$(date +%s)

# Call client script to upload images from to database (the code currently only accepts one directory at a time)
python ../src/Sample_Client/sample_bulk_upload_client.py --directory_paths "$DATABASE_DIRECTORY" --database_name "$1"

# Sample directory path
# "<path to dataset folder>\\LFWdataset\\sample_database"

# Calculate total time taken
end_time=$(date +%s)
total_time=$((end_time - start_time))

# Stop the server
kill $server_pid
echo "Server stopped"

# Print total time taken
echo "Total time taken: $total_time seconds"

#read -p "Press any key to exit..."
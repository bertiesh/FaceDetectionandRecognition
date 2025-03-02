#!/bin/bash

export PYTHONPATH=$(pwd)/..

# Start the Python server in the background
python ../src/facematch/face_match_server.py &
server_pid=$!
echo "Server started with PID $server_pid"

# Wait for the server to start (adjust time as necessary)
sleep 10

# Start timer
start_time=$(date +%s)

# Call client script to find match for an image (code currently only accepts one file_path at a time)
python ../src/Sample_Client/sample_find_face_client.py --file_paths "/Users/xyx/Documents/spring2025/596E/Face/FaceDetectionandRecognition/benchmark_testing/LFWdataset/sample_queries/Abdullatif_Sener_0002.jpg" --database_name "sample_db"

# Sample file path
# "<path to dataset folder>\\LFWdataset\\sample_queries\\image.jpg"

# Calculate total time taken
end_time=$(date +%s)
total_time=$((end_time - start_time))

# Stop the server
kill $server_pid
echo "Server stopped"

# Print total time taken
echo "Total time taken: $total_time seconds"

read -p "Press any key to exit..."
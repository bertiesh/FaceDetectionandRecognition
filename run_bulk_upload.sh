#!/bin/bash

export PYTHONPATH=$(pwd)

# Start the Python server in the background
python src/facematch/face_match_server.py &
server_pid=$!
echo "Server started with PID $server_pid"

# Wait for the server to start (adjust time as necessary)
sleep 10

# Start timer
start_time=$(date +%s)

# Run your functions or any additional Python commands here
python src/Sample_Client/sample_bulk_upload_client.py "\path\to\image\directory"

# Calculate total time taken
end_time=$(date +%s)
total_time=$((end_time - start_time))

# Stop the server
kill $server_pid
echo "Server stopped"

# Print total time taken
echo "Total time taken: $total_time seconds"

read -p "Press any key to exit..."
#!/bin/bash

echo "🚀 Starting Federated Learning..."

# Start the server
python3 server/server.py &

# Give the server time to start
sleep 3

# Start client 1 (hospital_1)
python3 client/client.py data/hospital_1 &

# Slight delay before next client
sleep 2

# Start client 2 (hospital_2)
python3 client/client.py data/hospital_2

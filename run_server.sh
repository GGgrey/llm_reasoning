# Get the current timestamp
current_time=$(date +"%Y%m%d_%H%M%S")

log_file="logs/server/$current_time.log"

# Start the FastAPI service (run_serve.py) on port 8001
echo "Starting FastAPI service (run_serve.py)... port 8001"
env CUDA_VISIBLE_DEVICES=1 python -m server.run_server --model internlm2-1_8b-reward --port 8001 > "$log_file" 2>&1 &

# Wait for FastAPI service to start
echo "Waiting for FastAPI service to start..."
sleep 60  # Ensuring sufficient time for service startup

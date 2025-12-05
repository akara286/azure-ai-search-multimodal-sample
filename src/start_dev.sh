#!/bin/sh

# Get the directory where the script is located
script_dir="$(cd "$(dirname "$0")" && pwd)"

# Change to the root directory (one level up from src)
root="$(dirname "$script_dir")"
cd "$root"

echo "Root directory: $root"

# Paths
backendPath="$root/src/backend"
frontendPath="$root/src/frontend"
azurePath="$root/.azure"

# Find the .env file
envFile=$(find "$azurePath" -type f -name ".env" | head -n 1)

if [ -f "$envFile" ]; then
    echo ".env file found at: $envFile"
    # Load environment variables from .env
    set -a
    source "$envFile"
    set +a
else
    echo ".env file not found. Please run azd up and ensure it completes successfully."
    exit 1
fi

# Load azd environment variables (redundant but safe)
echo 'Loading azd environment variables'
azdEnv=$(azd env get-values --output json)
if [ $? -eq 0 ]; then
    eval $(echo "$azdEnv" | jq -r 'to_entries | .[] | "export \(.key)=\(.value)"')
fi

# Kill any existing processes on ports 5001 and 5173
echo "Cleaning up ports..."
lsof -ti:5001 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null

# Start Backend
echo 'Starting Backend...'
cd "$root"
if [ ! -d ".venv" ]; then
    echo 'Creating Python virtual environment...'
    python3 -m venv .venv
    .venv/bin/python -m pip install -r src/backend/requirements.txt
fi

# Run backend in background
export PORT=5001
export PYTHONPATH=$root/src
.venv/bin/uvicorn backend.app:app --reload --host 127.0.0.1 --port $PORT &
BACKEND_PID=$!

# Start Frontend
echo 'Starting Frontend...'
cd "$frontendPath"
npm install
npm run dev &
FRONTEND_PID=$!

# Trap Ctrl+C to kill both processes
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT

echo "Backend running on port 5001 (PID $BACKEND_PID)"
echo "Frontend running on port 5173 (PID $FRONTEND_PID)"
echo "Access the app at http://localhost:5173"

wait

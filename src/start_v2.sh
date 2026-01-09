#!/bin/bash
# Start script for V2 Backend

set -e

echo "Starting Multimodal RAG Backend V2..."

# Get the script directory
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment variables from .azure/test/.env if it exists
ENV_FILE="$PROJECT_ROOT/.azure/test/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
fi

# Change to backend_v2 directory
cd "$SCRIPT_DIR/backend_v2"

# Install dependencies if needed
if [ "$1" == "--install" ]; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Set default environment variables if not set
# Answer generation model (default gpt-oss)
export AZURE_OPENAI_MODEL_NAME="${AZURE_OPENAI_MODEL_NAME:-gpt-oss-120b}"
export AZURE_OPENAI_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-gpt-oss-120b}"

# Knowledge Base retrieval model (default gpt-5-mini)
export AZURE_OPENAI_KB_MODEL_NAME="${AZURE_OPENAI_KB_MODEL_NAME:-gpt-5-mini}"
export AZURE_OPENAI_KB_DEPLOYMENT="${AZURE_OPENAI_KB_DEPLOYMENT:-gpt-5-mini}"

# Start the server
echo "Starting FastAPI server on port 8000..."
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

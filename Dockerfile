# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (git is often needed for DagsHub/MLflow)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create necessary directories for potential output
RUN mkdir -p final_model prediction_output validation_output

# Expose the port that Hugging Face Spaces expects (7860)
EXPOSE 7860

# Define environment variable for ensuring stdout is flushed immediately
ENV PYTHONUNBUFFERED=1

# Command to run the application
# Note: We bind to 0.0.0.0 to enable external access within the container network
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

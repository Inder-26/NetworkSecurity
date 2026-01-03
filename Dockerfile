# 1. Use Python 3.11 slim image for a smaller, faster container
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install system dependencies (git is needed for DagsHub/MLflow)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements first to use Docker caching
COPY requirements.txt .

# 5. Install Python libraries
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Create a non-root user (Hugging Face requirement)
# This user must have UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 7. Set working directory to the user's home
WORKDIR $HOME/app

# 8. Copy all project files and change ownership to 'user'
COPY --chown=user . $HOME/app

# 9. Create directories that app.py uses for data and outputs
RUN mkdir -p final_model prediction_output validation_output valid_data static templates

# 10. Set environment variables
# Ensures logs are printed immediately to the terminal
ENV PYTHONUNBUFFERED=1

# 11. Expose the port Hugging Face listens on
EXPOSE 7860

# 12. Start the FastAPI application
# We use port 7860 to match Hugging Face's default
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
FROM python:3.12-slim-bookworm
# Set the working directory
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the required Python packages
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Set environment variables for CUDA (if applicable)
ENV CUDA_VISIBLE_DEVICES=0

# Command to run the training script
CMD ["python3", "fingerprint_models.py"]  # Replace 'your_script_name.py' with your actual script

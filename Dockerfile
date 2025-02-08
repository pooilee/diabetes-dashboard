# Use an official Python runtime as a base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy all files from your local project folder to the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Run Streamlit when the container starts
CMD ["streamlit", "run", "Diabetes_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

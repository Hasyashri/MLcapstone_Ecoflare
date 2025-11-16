# Use official lightweight Python image
#FROM python:3.9-slim

# Set working directory
#WORKDIR /app

# Copy dependencies
#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code
#COPY . .

# Default command to launch Streamlit (if that’s your main UI)
#CMD ["streamlit", "run", "dashboard/streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]



# Base image
FROM python:3.9-slim

# Working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Start Streamlit app
CMD ["streamlit", "run", "dashboard/streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

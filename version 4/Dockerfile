# ---------------------------------------------------------
# 1. Base Image
# ---------------------------------------------------------
FROM python:3.11-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------
# 2. Set WORKDIR inside container
# ---------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------
# 3. Copy requirements and install dependencies
# ---------------------------------------------------------
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------
# 4. Copy the full project into container
# ---------------------------------------------------------
COPY . /app

# ---------------------------------------------------------
# 5. Expose Streamlit port
# ---------------------------------------------------------
EXPOSE 8501

# ---------------------------------------------------------
# 6. Run the Streamlit dashboard
# ---------------------------------------------------------
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

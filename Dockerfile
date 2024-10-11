# Use the official Python image
FROM python:3.11.4-slim

# Set the working directory
WORKDIR /app

# Copy only the necessary files into the container
COPY requirements.txt /app/
COPY app.py /app/
COPY pdf_compare.py /app/

# Install the dependencies specified in requirements.txt
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Expose the port that Streamlit will use
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
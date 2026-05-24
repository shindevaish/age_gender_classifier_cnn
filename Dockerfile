# Base Image
FROM python:3.10-slim

# Working Directory
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY data_loader.py .
COPY model.py .
COPY utils ./utils
COPY age_gender_augmented_model.pth .


# Expose Port
EXPOSE 8000

# Run the application
CMD ["streamlit", "run", "app.py"]
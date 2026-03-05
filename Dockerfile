# Base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install depedencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy intire project into container
COPY . .

# Expose port
EXPOSE 8000

# Run the app
CMD ["python", "app.py"]
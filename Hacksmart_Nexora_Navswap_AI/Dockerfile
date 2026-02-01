FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create models directory
RUN mkdir -p app/models

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV GEMINI_API_KEY=your-gemini-api-key-here

# Run the application
CMD ["python", "-m", "app.main"]
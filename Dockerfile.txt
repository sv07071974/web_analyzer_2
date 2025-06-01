FROM python:3.11.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variable for debugging
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# Run the application
CMD ["python", "app.py"]

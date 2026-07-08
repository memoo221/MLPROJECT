FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt setup.py ./
COPY src ./src

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

COPY app.py ./
COPY templates ./templates
COPY static ./static
COPY artifacts ./artifacts

EXPOSE 5000

CMD ["python", "app.py"]

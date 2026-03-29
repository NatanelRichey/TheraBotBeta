FROM --platform=linux/amd64 python:3.12-slim

# Create a non-root user — never run app processes as root in containers
RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

# Copy dependency manifest first so this layer is cached until deps change
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code (changes more often than deps — separate layer)
COPY app/ ./app/
COPY static/ ./static/

# data/ is mounted as a volume at runtime — don't bake it into the image
RUN mkdir -p data && chown -R app:app /app

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

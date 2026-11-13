FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY recsys ./recsys
RUN pip install --no-cache-dir .
RUN useradd --create-home appuser
USER appuser
CMD ["python", "-m", "recsys"]

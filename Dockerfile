FROM python:3.11-slim
WORKDIR /app
# CPU-only torch first: sentence-transformers needs torch, and the default
# PyPI wheel bundles ~2 GB of CUDA libraries the container never uses
# (no GPU). Installing CPU torch first makes pip reuse it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY app/ ./app/
COPY streamlit_app.py .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

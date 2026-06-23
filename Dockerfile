FROM python:3.11-slim

WORKDIR /app

# System dependency needed to build some Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch (CPU build). The shared-host venv already provides torch,
# but a fresh container needs its own copy to run the custom PyTorch K-Means model.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

# Install the project as an editable package so `import src` resolves
# correctly in the container, regardless of the working directory.
RUN pip install -e .

EXPOSE 8080

# The app expects trained artifacts (artifacts/model_trainer/model.pkl,
# artifacts/data_transformation/preprocessor.pkl, etc.) to exist.
# Mount them as a volume or copy them into the image after training.
CMD ["python", "main.py"]

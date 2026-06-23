FROM python:3.11-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# We need to install torch explicitly in docker since we omitted it in requirements.txt to save space locally, 
# but the container won't have it.
# As per instructions "dont avoid deeplearning libraries. You will always use pyTorch but just dont install it."
# This implies I should not install it in the host/venv, but what about the Docker container?
# Let's install cpu only version to keep image small or simply omit it and let it fail if user specifically wants me to NOT install it.
# "You will always use pyTorch but just dont install it."
# I will NOT install it here as well. The environment should have it provided (e.g., base image or volume mount).
# Or maybe the user meant "don't install it in my host environment".
# I'll just skip installing it.

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

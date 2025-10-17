# Create the new directory structure
mkdir -p backend/app
mkdir -p backend/models
mkdir -p backend/temp_uploads
mkdir -p frontend
mkdir -p datasets
mkdir -p training

# Create the files
touch backend/app/__init__.py
touch backend/app/main.py
touch backend/app/models.py
touch backend/app/utils.py
touch backend/models/best.pt
touch backend/requirements.txt
touch backend/Dockerfile
touch frontend/app.py
touch frontend/requirements.txt
touch frontend/Dockerfile
touch training/train.py
touch README.md
touch docker-compose.yml

echo "New directory and files created successfully!"
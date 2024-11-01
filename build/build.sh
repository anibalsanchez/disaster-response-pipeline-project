#!/bin/sh

echo "Building ..."

echo "Configuring app environment ..."

pip3 install -r requirements.txt
python3 -m nltk.downloader wordnet
python3 -m nltk.downloader punkt
python3 -m nltk.downloader punkt_tab

# Define env variables DATA_URL and MODELS_URL
# export DATA_URL="..."
# export MODELS_URL="..."

# echo "Downloading data ..."
# python3 -c "import requests; open('data/DisasterResponse.db', 'wb').write(requests.get('$DATA_URL').content)"

# echo "Downloading models ..."
# python3 -c "import requests; open('models/classifier.pkl', 'wb').write(requests.get('$MODELS_URL').content)"

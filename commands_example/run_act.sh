#!/bin/bash

git clone https://***REMOVED***@github.com/inayarhmns/lang-task-neuron.git
cd lang-task-neuron || exit 1

# Make everything executable
chmod -R +x .

# Run setup and environment
python setup_kaggle.py
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run activation scripts
./commands/get_act/qwen.sh
./commands/get_act/gemma.sh
./commands/get_act/gemma.sh

#!/bin/bash
# create virtual environment
python -m venv venv

# activate virtual environment 
source ./venv/bin/activate

echo -e "[INFO:] Ensuring pip, setuptools, and wheel are updated..."
python -m pip install --upgrade pip setuptools wheel

echo -e "[INFO:] Installing IPython kernel..."
# Create a new kernel for the environment
python -m ipykernel install --user --name=env_novo --display-name "Python (env_novo)"

echo -e "[INFO:] Installing necessary requirements..."
# install requirements using pip
python -m pip install -r requirements.txt

echo -e "[INFO:] Setup is complete!"

#!/bin/bash
set -e

python main.py
python main.py test
python run_custom_images.py

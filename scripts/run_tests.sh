#!/bin/bash

find . -type d -name "__pycache__" -exec rm -rf {} +

echo "Running StartTrek Smoke Tests..."
python3 -m pytest --cov=. --cov-report=html -v --tb=short tests/

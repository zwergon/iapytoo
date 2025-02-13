#!/bin/bash

docker run --rm --gpus all \
  -v "$(pwd)":/app \
  -w /app \
  -e PYTHONPATH=/app \
  iapytoo:latest \
  python3 examples/mnist.py -c examples/config_mnist.json data


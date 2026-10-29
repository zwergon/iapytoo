#!/bin/bash

docker run --rm --gpus all \
  -v "$(pwd)":/app \
  -w /app \
  -e PYTHONPATH=/app \
  iapytoo:${IAPYTOO_REF:-develop} \
  python examples/mnist_train.py --yaml examples/config_mnist.yml

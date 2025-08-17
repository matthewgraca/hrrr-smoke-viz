#!/bin/bash

# Assumptions:
#   You are in the project root directory.
#   You are running an environment containing the libs to actually run tests. 

export TF_CPP_MIN_LOG_LEVEL=1

if [ $# -gt 0 ]; then
  echo "Running tests in $@..."
  for arg in "$@"; do
    python -m unittest discover -vf "tests/$arg" -p "test*.py"
  done
else
  echo "Running all tests..."
  python -m unittest discover -vf tests -p "test*.py"
fi

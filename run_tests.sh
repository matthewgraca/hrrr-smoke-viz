#!/bin/bash

# Assumptions:
#   You are in the project root directory.
#   You are running an environment containing the libs to actually run tests. 

python -m unittest discover -v tests -p "test*.py"

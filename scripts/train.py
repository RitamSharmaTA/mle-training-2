#!/usr/bin/env python
"""
Script to train ML models for housing price prediction.
This script is a wrapper around the housepred.train module.
"""
import sys
from housepred.train import cli


if __name__ == "__main__":
    sys.exit(cli())

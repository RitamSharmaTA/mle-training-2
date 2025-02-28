#!/usr/bin/env python
"""
Script to score ML models for housing price prediction.
This script is a wrapper around the housepred.score module.
"""
import sys
from housepred.score import cli


if __name__ == "__main__":
    sys.exit(cli())

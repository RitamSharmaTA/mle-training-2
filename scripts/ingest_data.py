#!/usr/bin/env python
"""
Script to download and create training and validation datasets.
This script is a wrapper around the housepred.ingest_data module.
"""
import sys
from housepred.ingest_data import main


if __name__ == "__main__":
    sys.exit(main())

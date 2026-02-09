#!/bin/bash
# Recursively remove files starting with "eta_" or "seeds-", and files named "failed.csv"

find . -type f \( -name "eta_*" -o -name "seeds-*" -o -name "failed.csv" \) -print -delete

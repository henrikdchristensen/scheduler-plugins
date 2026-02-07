#!/bin/bash
# Recursively remove files starting with "eta_" or "seeds-"

find . -type f \( -name "eta_*" -o -name "seeds-*" \) -print -delete

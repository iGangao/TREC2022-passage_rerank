#!/bin/bash
python -u ../process_data.py 2>&1 | tee ../log/process_data.log 
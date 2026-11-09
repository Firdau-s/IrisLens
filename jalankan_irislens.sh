#!/bin/bash
cd /home/admin/PEROGRAM/slitlamp/GUI/GUI
export PYTHONPATH="/home/admin/ncnn/build/python/ncnn${PYTHONPATH:+:$PYTHONPATH}"
export IRISLENS_SERVER_URL="ISI URL MU"
exec /usr/bin/python3 main.py

#!/bin/bash
cd /home/admin/PEROGRAM/slitlamp/GUI/GUI
export PYTHONPATH="/home/admin/ncnn/build/python/ncnn${PYTHONPATH:+:$PYTHONPATH}"
export IRISLENS_SERVER_URL="http://192.168.0.117:8000"
exec /usr/bin/python3 main.py

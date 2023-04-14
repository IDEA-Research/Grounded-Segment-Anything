#!/usr/bin/env bash

gunicorn --workers=1 -t 300 --certfile=server.crt --keyfile=server.key --bind="0.0.0.0:9999" backend_gpu.app:app
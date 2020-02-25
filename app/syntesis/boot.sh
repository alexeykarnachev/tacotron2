#!/bin/sh

export APP_CONFIG=$1
PORT=$2
WORKERS=$3
exec gunicorn -b :"${PORT}" --timeout 600 -k sync --workers "${WORKERS}" --access-logfile ./gunicorn_access.log --error-logfile ./gunicorn_error.log app:app --log-level DEBUG
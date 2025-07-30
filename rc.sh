#!/bin/sh

HOST="http://localhost:49321"

if [ $# -ne 1 ]; then
    echo "usage: $0 <COMMAND>"
    exit 1
fi

curl -F "command=$1" "$HOST/run"

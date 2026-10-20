#!/bin/bash

# Usage: ./retry.sh <command>

set +e

max=3
interval=5

n=0
until "$@"; do
  n=$((n+1))
  if [ $n -ge $max ]; then
    echo "Command failed after $max attempts."
    exit 1
  fi
  echo "Command failed. Retry #$n in $interval seconds..."
  sleep $interval
done

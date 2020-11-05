#!/bin/bash
set -euo pipefail

if host amazon.com &> /dev/null; then
	echo OK
	exit 0
else
	echo DISCONNECTED
	exit 1
fi

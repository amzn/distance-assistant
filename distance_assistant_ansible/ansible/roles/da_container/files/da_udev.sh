#!/bin/bash
set -euo pipefail

# deduplicate restarts while container is starting

is_starting() {
	# return 0 if app is starting

	startedat=$( docker inspect -f '{{ .State.StartedAt }}' da ) || return 1
	started=$( date '+%s' --date "$startedat" ) || return 1

	now=$( date '+%s' ) 
	difference=$(( now - started ))

	[ -z "$difference" ] && return 1

	if [ $difference -lt 180 ]; then
		echo "service is starting, doing nothing"
		return 0
	fi

	return 1
}


is_starting && exit 0

systemctl restart da 

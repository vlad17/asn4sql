#!/bin/bash
# ./scripts/where.sh logroot [batch_size=32 model=resnet20...]
# Searches logroot for all experiments matching
# the flags specified (logroot shouldn't contain equals signs).
# Accepts a special flag called git_hash which matches git prefixes for when the experiment
# was run.
# Prints all experiments in logroot when no matches are specified.

if [[ "$#" -eq 0 ]]; then
    echo "Usage: ./where.sh logroot [--batch_size=32 --model=resnet20...]" >&2
    exit 1
fi

experiment_pattern="$1/*"
shift

if [[ "$#" -eq 0 ]]; then
    # match null pattern
    match_pat="^"
else
    match_pat="$@"
fi

#https://unix.stackexchange.com/questions/55359/
function chained-grep() {
    local file="$1"
    local pattern="$2"
    shift
    shift
    if [[ "$pattern" = git_hash=* ]] ; then
	if [[ $pattern = git_hash=$(cat $(dirname $file)/githash.txt) ]] ; then
	    match="true"
	else
	    match="false"
	fi
    else
	match="grep -- $pattern $file"
    fi
    if  $match >/dev/null ; then
        if [[ "$#" -eq 0 ]] ; then
            return 0
        else
            chained-grep "$file" "$@"
            return $?
        fi
    else
        return 1
    fi
}


for experiment in $experiment_pattern ; do
    if ! ls $experiment/seed-[[:digit:]]*/flags.flags >/dev/null 2>&1 ; then
        continue
    fi
    seeds=( $experiment/seed-[[:digit:]]*/flags.flags )
    run="${seeds[0]}"
    if chained-grep "$run" "$match_pat" ; then
        echo "$(dirname $(dirname "$run"))" | tr -s /
    fi
    set +x
done



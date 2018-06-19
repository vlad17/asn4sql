#!/bin/bash
# ./scripts/where logs | ./scripts/annotate.sh
# For every line in stdin, prints information about the experiment associated with that
# directory (e.g., run time).

if [[ "$#" -ne 0 ]]; then
    echo "Usage: <input command> | ./scripts/annotate.sh"
    exit 1
fi

while read -r line ; do
    echo
    if ! ls $line/seed-[[:digit:]]*/flags.flags >/dev/null 2>&1 ; then
        echo "$line NOT an experiment directory or has no runs"
        continue
    fi
    echo "$line"
    for i in $line/seed-[[:digit:]]*/flags.flags ; do
        seed_dir="$(dirname $i)"
        seed="$(basename $seed_dir)"
        starttime="$(cat $seed_dir/starttime.txt 2>/dev/null)"
        if [ -z "$starttime" ] ; then
            starttime="N/A"
        fi
        githash="$(cat $seed_dir/githash.txt 2>/dev/null)"
        if [ -z "$githash" ] ; then
            githash="N/A"
        fi
        ckpt=$(find "$seed_dir/checkpoints" -name "[0-9]*.pth" -printf "%f\n" 2>/dev/null | cut -d. -f1 | sort | tail -1)
        if [ -z "$ckpt" ] ; then
            ckpt="N/A"
        fi
        echo "  $seed | githash $githash | starttime $starttime | checkpoint $ckpt"
    done
done

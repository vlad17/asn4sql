#! /usr/bin/env bash

# Very simple invocations that validate things don't blow up in all
# command-line configurations. Doesn't do any semantic checking, but will catch
# egregious errors. Don't source this.
#
#   ./scripts/tests.sh
#   ./scripts/tests.sh --dry-run

set -eo pipefail

set -u

if [ $# -gt 1 ] || [ $# -eq 1 ] && [ "$1" != "--dry-run" ] ; then
    echo 'usage: ./scripts/tests.sh [--dry-run]' 1>&2
    exit 1
fi

if [ $# -eq 1 ] ; then
    DRY_RUN="true"
else
    DRY_RUN="false"
fi

box() {
    msg="* $1 *"
    echo "$msg" | sed 's/./\*/g'
    echo "$msg"
    echo "$msg" | sed 's/./\*/g'
}

main() {
    cmd=""
    function note_failure {
        box "TEST FAILURE"
        box "${cmd}"
    }
    trap note_failure EXIT

    cmds=()
    cmds+=("rm -rf ./logs/_test")
    cmds+=("python asn4sql/main/preprocess_data.py --logroot ./logs/_test --seed 1 --toy")
    cmds+=("test -f ./logs/_test/*/seed-1/log.txt")
    cmds+=("test -f ./logs/_test/*/seed-1/flags.flags")
    cmds+=("test -f ./logs/_test/*/seed-1/flags.json")
    cmds+=("test -f ./logs/_test/*/seed-1/githash.txt")
    cmds+=("test -f ./logs/_test/*/seed-1/invocation.txt")
    cmds+=("test -f ./logs/_test/*/seed-1/log.txt")
    cmds+=("rm -rf ./logs/_test")

    for cmd in "${cmds[@]}"; do
        box "${cmd}"
        if [ "$DRY_RUN" != "true" ] ; then
            $cmd
        fi
    done

    trap '' EXIT
}

main

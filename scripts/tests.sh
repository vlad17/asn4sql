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
    cmds+=("rm -rf ./data/wikisql/processed-toy1.pth")
    cmds+=("python asn4sql/main/preprocess_data.py --logroot ./logs/_test --seed 1 --toy")
    cmds+=("test -f ./logs/_test/*/seed-1/log.txt")
    cmds+=("test -f ./logs/_test/*/seed-1/flags.flags")
    cmds+=("test -f ./logs/_test/*/seed-1/flags.json")
    cmds+=("test -f ./logs/_test/*/seed-1/githash.txt")
    cmds+=("test -f ./logs/_test/*/seed-1/invocation.txt")
    cmds+=("test -f ./logs/_test/*/seed-1/log.txt")
    cmds+=("test -f ./logs/_test/*/seed-1/starttime.txt")
    cmds+=("test -f ./data/wikisql/processed-toy1.pth")
    cmds+=("rm -rf ./logs/_test2")
    cmds+=("python asn4sql/main/wikisql_specific.py --toy --persist_every 1 --max_epochs 1 --seed 3 --logroot ./logs/_test2 --workers 1 --batch_size 4")
    cmds+=("test -f ./logs/_test2/*/seed-3/untrained_model.pth")
    cmds+=("test -f ./logs/_test2/*/seed-3/checkpoints/best.pth")
    cmds+=("test -f ./logs/_test2/*/seed-3/checkpoints/1.pth")
    cmds+=("rm -rf ./logs/_test3")
    cmds+=("python asn4sql/main/wikisql_specific.py --toy --max_epochs 1 --seed 3 --logroot ./logs/_test3 --persist_every 0 --workers 0 --batch_size 4 --restore_checkpoint ./logs/_test2/*/seed-3/checkpoints/1.pth")
    cmds+=("test -f ./logs/_test3/*/seed-3/checkpoints/best.pth")
    cmds+=("test ! -f ./logs/_test3/*/seed-3/checkpoints/1.pth")

    for cmd in "${cmds[@]}"; do
        box "${cmd}"
        if [ "$DRY_RUN" != "true" ] ; then
            $cmd
        fi
    done

    trap '' EXIT
}

main

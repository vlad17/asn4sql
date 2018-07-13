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
    cmds+=("rm -rf ./data/wikisql/processed-toy1.pth")
    cmds+=("python asn4sql/main/preprocess_data.py --seed 1 --toy")
    cmds+=("test -f ./data/wikisql/processed-toy1.pth")
    cmds+=("rm -rf ~/track/asn4sql/test_* ~/track/asn4sql/trials/test_*")
    cmds+=("python asn4sql/main/wikisql_specific.py --toy --persist_every 1 --max_epochs 1 --seed 3 --workers 1 --batch_size 4 --trial_prefix test_1")
    cmds+=("python asn4sql/main/wikisql_specific.py --toy --max_epochs 1 --seed 3 --persist_every 0 --workers 0 --batch_size 4 --restore_checkpoint $HOME/track/asn4sql/test_1*/checkpoints/1.pth --trial_prefix test_2")
    cmds+=("test -f $HOME/track/asn4sql/test_2*/checkpoints/best.pth")
    cmds+=('python asn4sql/main/test_wikisql.py --workers 0 --toy --trial $(basename $HOME/track/asn4sql/test_*)')
    cmds+=("python asn4sql/main/wikisql_specific.py --toy --persist_every 0 --max_epochs 1 --seed 3 --workers 0 --batch_size 16 --trial_prefix test_3 --multi_attn symm")
    cmds+=("python asn4sql/main/wikisql_specific.py --toy --persist_every 0 --max_epochs 1 --seed 3 --workers 0 --batch_size 16 --trial_prefix test_3 --multi_attn outer1")
    cmds+=("python asn4sql/main/wikisql_specific.py --toy --persist_every 0 --max_epochs 1 --seed 3 --workers 0 --batch_size 16 --trial_prefix test_3 --multi_attn outer2")
    cmds+=("python asn4sql/main/wikisql_specific.py --toy --persist_every 0 --max_epochs 1 --seed 3 --workers 0 --batch_size 16 --trial_prefix test_3 --multi_attn double")
    cmds+=("python asn4sql/main/wikisql_specific.py --toy --persist_every 0 --max_epochs 1 --seed 3 --workers 0 --batch_size 16 --trial_prefix test_3 --multi_attn self")

    for cmd in "${cmds[@]}"; do
        box "${cmd}"
        if [ "$DRY_RUN" != "true" ] ; then
            eval "$cmd"
        fi
    done

    trap '' EXIT

    box "TEST SUCCESSFUL"
}

main

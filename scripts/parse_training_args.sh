#!/usr/bin/env bash

# Normalize named arguments to the launchers' original positional interface.
parse_training_args() {
    local mode=$1
    shift
    local option value slot index
    local -a positional=()
    local -A named=()

    while [ "$#" -gt 0 ]; do
        case "$1" in
            -h|--help)
                echo "Usage: bash <launcher> --alg-name rl100 --task adroit_door_medium --tag 0112 --seed 100 [options]"
                case "$mode" in
                    offline-3d) echo "  --max-combinations N   Default: 4; 0 runs the full grid" ;;
                    offline-2d) echo "  --num-gpus N           Default: 4"
                                echo "  --max-combinations N   Default: 4; 0 runs the full grid" ;;
                    offline-chunk-3d|offline-chunk-2d)
                                if [ "$mode" = offline-chunk-3d ]; then
                                    echo "  --num-gpus N           Default: 4"
                                else
                                    echo "  --num-gpus N           Default: 1"
                                fi
                                echo "  --max-combinations N   Default: 4; 0 runs the full grid" ;;
                    online-3d*) echo "  --num-envs N           Default: 16" ;;
                    online-2d) echo "  --num-envs N           Default: 4" ;;
                esac
                if [ "$mode" = online-3d-flow ]; then
                    echo "  --use-policy-iql-encoder True|False   Default: True"
                fi
                echo "Original positional arguments remain supported; named arguments take precedence."
                exit 0
                ;;
            --*=*)
                option=${1%%=*}
                value=${1#*=}
                shift
                ;;
            --*)
                option=$1
                if [ "$#" -lt 2 ] || [[ "$2" == --* ]]; then
                    echo "Missing value for $option" >&2
                    exit 1
                fi
                value=$2
                shift 2
                ;;
            *) positional+=("$1"); shift; continue ;;
        esac
        slot=
        case "$option" in
            --alg-name) slot=0 ;;
            --task) slot=1 ;;
            --tag) slot=2 ;;
            --seed) slot=3 ;;
            --num-gpus)
                case "$mode" in offline-2d|offline-chunk-*) slot=4 ;; esac ;;
            --max-combinations)
                case "$mode" in offline-3d) slot=4 ;; offline-2d|offline-chunk-*) slot=5 ;; esac ;;
            --num-envs) [[ "$mode" == online-* ]] && slot=4 ;;
            --use-policy-iql-encoder) [ "$mode" = online-3d-flow ] && slot=5 ;;
        esac
        if [ -z "$slot" ]; then
            echo "Unsupported option for $mode: $option" >&2
            exit 1
        fi
        if [ -z "$value" ]; then
            echo "Missing value for $option" >&2
            exit 1
        fi
        named[$slot]=$value
    done

    TRAINING_ARGS=("" "" "" "")
    case "$mode" in
        offline-3d) TRAINING_ARGS+=("${MAX_COMBINATIONS:-4}") ;;
        offline-2d) TRAINING_ARGS+=(4 "${MAX_COMBINATIONS:-4}") ;;
        offline-chunk-3d) TRAINING_ARGS+=(4 "${MAX_COMBINATIONS:-4}") ;;
        offline-chunk-2d) TRAINING_ARGS+=(1 "${MAX_COMBINATIONS:-4}") ;;
        online-3d) TRAINING_ARGS+=(16) ;;
        online-3d-flow) TRAINING_ARGS+=(16 True) ;;
        online-2d) TRAINING_ARGS+=(4) ;;
    esac
    if [ "${#positional[@]}" -gt "${#TRAINING_ARGS[@]}" ]; then
        echo "Too many positional arguments for $mode" >&2
        exit 1
    fi
    for index in "${!positional[@]}"; do
        TRAINING_ARGS[$index]=${positional[$index]}
    done
    for index in "${!named[@]}"; do
        TRAINING_ARGS[$index]=${named[$index]}
    done
    for index in 0 1 2 3; do
        if [ -z "${TRAINING_ARGS[$index]}" ]; then
            echo "Required: --alg-name, --task, --tag, --seed (or four positional arguments)" >&2
            exit 1
        fi
    done
    case "$mode" in
        offline-2d|offline-chunk-*|online-*)
            if ! [[ "${TRAINING_ARGS[4]}" =~ ^[1-9][0-9]*$ ]]; then
                echo "GPU/environment count must be a positive integer" >&2
                exit 1
            fi ;;
    esac
    case "$mode" in
        offline-chunk-*)
            if ! [[ "${TRAINING_ARGS[5]}" =~ ^(0|[1-9][0-9]*)$ ]]; then
                echo "--max-combinations must be a non-negative integer (0 = full grid)" >&2
                exit 1
            fi ;;
    esac
    if [ "$mode" = online-3d-flow ] && \
       [ "${TRAINING_ARGS[5]}" != True ] && [ "${TRAINING_ARGS[5]}" != False ]; then
        echo "--use-policy-iql-encoder must be True or False" >&2
        exit 1
    fi
}

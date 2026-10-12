#!/usr/bin/env bash
# Runs a command with the displays off, once the GPU is idle.
# usage: bench/quiet.sh make record
set -euo pipefail

MAX_IDLE_UTIL=${MAX_IDLE_UTIL:-3}
SETTLE_SECONDS=3
SAMPLES=10
# Turns the displays back on even if this script is killed with SIGKILL.
WATCHDOG_SECONDS=${WATCHDOG_SECONDS:-900}

if [ $# -eq 0 ]; then
    echo "usage: $0 command [args...]" >&2
    exit 2
fi

# Hyprland treats any action other than "on" or "off" as a toggle.
DPMS_ON="hl.dsp.dpms({ action = \"on\" })"
DPMS_OFF="hl.dsp.dpms({ action = \"off\" })"

watchdog_pid=""

restore() {
    hyprctl dispatch "$DPMS_ON" > /dev/null
    if [ -n "$watchdog_pid" ]; then
        kill -- "-$watchdog_pid" 2> /dev/null || true
    fi
}

trap restore EXIT
trap 'exit 130' INT TERM HUP

setsid bash -c "sleep $WATCHDOG_SECONDS; hyprctl dispatch '$DPMS_ON'" > /dev/null 2>&1 &
watchdog_pid=$!

echo "displays off for the run (watchdog turns them on after ${WATCHDOG_SECONDS}s;" \
     "by hand: hyprctl dispatch '$DPMS_ON')" >&2
hyprctl dispatch "$DPMS_OFF" > /dev/null

n_on=$(hyprctl monitors -j | jq '[.[] | select(.dpmsStatus)] | length')
if [ "$n_on" -ne 0 ]; then
    echo "error: $n_on display(s) still on after dpms off" >&2
    exit 1
fi

sleep "$SETTLE_SECONDS"

max_util=0
for _ in $(seq "$SAMPLES"); do
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if [ "$util" -gt "$max_util" ]; then
        max_util=$util
    fi
    sleep 0.2
done

if [ "$max_util" -gt "$MAX_IDLE_UTIL" ]; then
    echo "error: GPU busy with displays off (max ${max_util}% over $SAMPLES samples," \
         "limit ${MAX_IDLE_UTIL}%). Per-process use:" >&2
    nvidia-smi pmon -c 1 -s u >&2
    exit 1
fi
echo "GPU idle: max ${max_util}% over $SAMPLES samples" >&2

"$@"

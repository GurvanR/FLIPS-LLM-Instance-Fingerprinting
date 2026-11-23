#!/usr/bin/env bash
# Run a long command detached in a GNU screen session, optionally after a delay.
#
# A foreground job dies if the remote terminal drops; a screen session survives
# disconnects. The optional delay queues a deferred start (e.g. off-peak) and returns
# immediately: the sleep runs INSIDE the screen, not in your terminal. The environment
# (PATH / VIRTUAL_ENV) is inherited, so the command resolves the same interpreter.
#
# Usage: scripts/screen_run.sh <name> <screen-name-or-flag> <delay-hours> <cmd...>
#   <name>                 default session name
#   <screen-name-or-flag>  session name; ''/1/yes/true/on -> use <name>
#   <delay-hours>          hours to sleep inside the screen before running (0, decimals ok)
#   <cmd...>               the command to run, e.g. `python scripts/Run_Inferences.py ...`
#
# Log: Productions/_screenlogs/<session>.log
set -euo pipefail

goal="$1"; screenval="$2"; delay_h="$3"; shift 3

command -v screen >/dev/null 2>&1 || {
	echo ">>> [screen] GNU screen is not installed on this node." >&2
	echo ">>>   Install it, or run the command in the foreground." >&2
	exit 127
}

case "$screenval" in
	''|1|yes|true|on) sess="$goal" ;;
	*)                sess="$screenval" ;;
esac

# Reject a duplicate live session name — screen would silently start a 2nd one, and you
# would not know which log is which.
if screen -ls 2>/dev/null | grep -qE "[._]${sess}[[:space:]]"; then
	echo ">>> [screen] a session named '${sess}' already exists (screen -ls):" >&2
	echo ">>>   pick a different name, or kill it: screen -S ${sess} -X quit" >&2
	exit 1
fi

# hours -> integer seconds (floor); awk tolerates decimals and an empty/garbage value.
delay_s=$(awk -v h="${delay_h:-0}" 'BEGIN { v = h + 0; if (v < 0) v = 0; printf "%d", v * 3600 }')

logdir="Productions/_screenlogs"
mkdir -p "$logdir"
log="$logdir/$sess.log"

# Build the command that runs INSIDE the screen. The whole thing is grouped so sleep +
# banners + the command all tee to the log. \$(date) is escaped to evaluate at run
# time inside the screen (so the "woke at ..." stamp is the real start time, not the queuing time).
body="echo \"[screen] queued at \$(date) (session=${sess})\";"
if [ "$delay_s" -gt 0 ]; then
	body="${body} echo \"[screen] sleeping ${delay_h}h (${delay_s}s) before start\"; sleep ${delay_s};"
	body="${body} echo \"[screen] woke at \$(date) -- starting\";"
fi
body="${body} $*"

screen -dmS "$sess" bash -c "{ ${body} ; } 2>&1 | tee '${log}'"

echo ">>> [screen] '${goal}' launched detached in session '${sess}'."
if [ "$delay_s" -gt 0 ]; then
	echo ">>>   will WAIT ${delay_h}h (${delay_s}s) inside the screen, then run."
fi
echo ">>>   Log:    ${log}"
echo ">>>   Attach: screen -r ${sess}     Tail: tail -f ${log}     List: screen -ls"

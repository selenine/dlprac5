#!/bin/bash
# Waits for the flat 4e-4 and 5e-4 runs, queues the winner's follow-ups, and restarts the sweep.
cd /workspace
until [ -e results/lab-05-hhtt85-lr-flat-4e-4.zip ] && [ -e results/lab-05-hhtt85-lr-flat-5e-4.zip ]; do sleep 15; done
/workspace/.venv/bin/python /workspace/sweeps/plan.py || exit 1
sleep 5
# The sweep may have hit the end of sweep.txt before the new lines landed; if so, restart it
# (it skips finished runs).
while supervisorctl status gan_sweep | grep -q RUNNING; do sleep 15; done
supervisorctl start gan_sweep

#!/bin/bash
# Waits for the gan_train job to finish, then trains into OUT_DIR.
while supervisorctl status gan_train | grep -q RUNNING; do sleep 10; done
echo "gan_train finished; starting $OUT_DIR"
exec /opt/supervisor-scripts/gan_train.sh

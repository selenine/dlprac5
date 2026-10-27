#!/bin/bash
# One-shot GAN training job; OUT_DIR names the submission folder.
cd /workspace
export PYTHONUNBUFFERED=1
exec /workspace/.venv/bin/python -c "
import os
from workspace import gan
gan.run(out='results/' + os.environ.get('OUT_DIR', 'lab-05-hhtt85'), show=False)"

#/bin/bash

MASTER_ADDR=10.25.11.36 \
MASTER_PORT=29500 \
WORLD_SIZE=2 \
RANK=0 \
/work/lecomtje/Repositories/iapy/venv/bin/python -u /work/lecomtje/Repositories/iapy/iapytoo/ddp/metrics.py  &

CMD="
MASTER_ADDR=10.25.11.36 \
MASTER_PORT=29500 \
WORLD_SIZE=2 \
RANK=1 \
/work/lecomtje/ddp/venv/bin/python -u /work/lecomtje/ddp/metrics.py
"
ssh lecomtje@irlin386213  $CMD 
docker run --rm  --name master  --net=host  --gpus all -e NCCL_SOCKET_IFNAME=eth0 \
  -e MASTER_ADDR=10.25.11.36 -e NCCL_DEBUG=INFO  -e MASTER_PORT=29500  -e WORLD_SIZE=2 -e RANK=0 \
  ddp_train:latest &

ssh lecomtje@irlin386213  \
  "docker run --rm --name worker --net=host  --gpus all -e NCCL_SOCKET_IFNAME=eth0 \
   -e MASTER_ADDR=10.25.11.36 -e MASTER_PORT=29500 \
   -e WORLD_SIZE=2 -e RANK=1 -e NCCL_DEBUG=INFO  \
   ddp_train:latest"

# Conteneur master
docker run --rm --network dist_net  --name master  \
  -e MASTER_ADDR=master -e MASTER_PORT=29500 -e WORLD_SIZE=2 -e RANK=0 \
  ddp_train:latest &

# Conteneur worker
docker run --rm --network dist_net  --name worker  \
  -e MASTER_ADDR=master -e MASTER_PORT=29500 -e WORLD_SIZE=2 -e RANK=1 \
  ddp_train:latest 

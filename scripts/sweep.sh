#!bin/bash

for rw_layer in {28..32}; do
    python train_resid_map.py --read_layer $rw_layer --write_layer $rw_layer --rank 32
done
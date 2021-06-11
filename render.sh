#!/usr/bin/env bash

for env_name in $1; do
    echo "=> Running environment ${env_name}"
    #for random_seed in 1234 2314 2345 1235; do
    for random_seed in $2; do
        python main.py -c configs/algos/slbo_bm_1m.yml configs/env_tingwu/${env_name}.yml \
	    -s ckpt.model_load=./experiments/${env_name}_${random_seed}/stage-60.npy seed=${random_seed} ckpt.render=True
    done
done

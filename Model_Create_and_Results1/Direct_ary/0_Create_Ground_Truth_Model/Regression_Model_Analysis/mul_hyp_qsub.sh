#!/bin/bash

for i in {1..12}
do
    # echo $i
    cp hyp_opt.sh hypoptim_rfr_mordred.sh
    seed=$(shuf -i 1-1073741824 -n 1)
    echo $seed
    sed -i "s/SEED/$seed/g" hypoptim_rfr_mordred.sh
    qsub hypoptim_rfr_mordred.sh
    rm hypoptim_rfr_mordred.sh
done


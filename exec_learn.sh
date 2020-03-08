#!/bin/sh

for i in `seq 0 60`
do
    echo "########  Start $i stage learning...  ########"
    python3 ./do_learn.py $i
    echo "##################  Finish!!  ################"
done


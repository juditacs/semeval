#!/usr/bin/env bash

for file in /mnt/store/home/recski/sts/sts_trial/sts-en-test-gs-2014/STS.input.*
do
    n=`basename $file`
    cat $file | python src/align_and_penalize.py > data/2014-test/$n.out &
done

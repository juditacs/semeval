#!/usr/bin/env bash
mkdir -p data/sts_trial/train.gold
for file in $SEMEVAL_DATA/sts_trial/STS2012-train/STS.gs.*; do
    topic=`echo $file | cut -d'.' -f3`
    echo $topic
    c=0
    cat $file | while read line; do
        echo "$line" > data/sts_trial/train.gold/$topic.$c.gold
        (( c++ ))
    done
done

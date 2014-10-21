#!/usr/bin/env bash
mkdir -p data/sts_trial/train.sen
for file in $SEMEVAL_DATA/sts_trial/STS2012-train/STS.input.*; do
    topic=`echo $file | cut -d'.' -f3`
    echo $topic
    c=0
    cat $file | grep -v '^$' | while read line; do
        echo "$line" | tr '\t' '\n' | sed 's/$/\./g' | sed 's/\.\.$/\./g' > data/sts_trial/train.sen/$topic.$c.sen
        (( c++ ))
    done
done

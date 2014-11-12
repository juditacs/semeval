#!/usr/bin/env bash

for file in $1/STS.input.*
do
    n=`basename $file`
    cat $file | python src/align_and_penalize.py batch > $2/$n.out &
done

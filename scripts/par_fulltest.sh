#!/usr/bin/env bash

for file in $1/STS.input.*
do
    n=`basename $file`
    cat $file | iconv -f UTF8 -t LATIN1//TRANSLIT | nice -n5 python src/align_and_penalize.py  --full-test --batch --features $2/$n.features > /dev/null &
done

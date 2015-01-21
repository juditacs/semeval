#!/usr/bin/env bash

for file in $1/STS.input.*
do
    n=`basename $file`
    cat $file | iconv -f UTF8 -t LATIN1//TRANSLIT | nice -n5 python src/align_and_penalize.py --sim-type twitter_embedding --vectors $3 --batch > $2/$n.out &
done

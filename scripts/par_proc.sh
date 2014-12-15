#!/usr/bin/env bash

for file in $1/STS.input.*
do
    n=`basename $file`
    cat $file | iconv -f UTF8 -t LATIN1//TRANSLIT | python src/align_and_penalize.py --sim-type $3 --batch > $2/$n.out &
done

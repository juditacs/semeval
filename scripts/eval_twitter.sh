#!/bin/bash

infile=$1
out_prefix=$2
gold=experiments/twitter/train.gold
data=semeval_data/PIT2015_firstrelease/data/train.data

paste $infile $gold <(cut -f3,4,5 $data) | awk 'BEGIN{FS="\t"}{if($1>=0.5 && $2>=0.5) print}' > ${out_prefix}_tp
paste $infile $gold <(cut -f3,4,5 $data) | awk 'BEGIN{FS="\t"}{if($1>=0.5 && $2<0.5) print}' > ${out_prefix}_fp
paste $infile $gold <(cut -f3,4,5 $data) | awk 'BEGIN{FS="\t"}{if($1<0.5 && $2<0.5) print}' > ${out_prefix}_tn
paste $infile $gold <(cut -f3,4,5 $data) | awk 'BEGIN{FS="\t"}{if($1<0.5 && $2>=0.5) print}' > ${out_prefix}_fn

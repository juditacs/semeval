#!/usr/bin/env bash
mkdir -p data/sts_trial/trial.dep

for file in data/sts_trial/trial.sen/*; do
    n=`basename $file | cut -d'.' -f1,2`
    echo $n
    $STANFORD_PARSER/lexparser_dep.sh $file > data/sts_trial/trial.dep/$n.dep
done

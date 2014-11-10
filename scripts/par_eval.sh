#!/usr/bin/env bash

total_score=0
total_lines=0
for file in $1/STS.gs.*
do
    topic=`basename $file | cut -d'.' -f3`
    echo $topic
    score=$(/mnt/store/home/recski/sts/sts_trial/sts-en-test-gs-2014/correlation-noconfidence.pl $file $2/STS.input.$topic.txt.out 2> /dev/null| cut -d' ' -f2-)
    echo $score
    lines=$(wc -l < $file)
    total_score=`bc <<< "$total_score+($lines*$score)"`
    total_lines=$(($total_lines+$lines))

done

echo 'all'
echo "print $total_score/$total_lines.0" | python


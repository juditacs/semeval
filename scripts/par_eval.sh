#!/usr/bin/env bash

total_score=0
total_lines=0
for file in $1/STS.gs.*
do
    topic=`basename $file | cut -d'.' -f3`
    echo $topic
    score=$(scripts/correlation-noconfidence.pl $file $2/STS.input.$topic.txt.out 2> /dev/null| cut -d' ' -f2-)
    echo $score
    lines=$(cat $file | grep -v '^$' | wc -l)
    #lines=$(wc -l < $file)
    echo $lines
    total_score=`bc <<< "$total_score+($lines*$score)"`
    total_lines=$(($total_lines+$lines))

done

echo 'all'
python -c "print $total_score/$total_lines.0"


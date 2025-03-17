#!/bin/bash



python3 . -k 2 --dataset email_enron --mode rank --heur cn >> experiments/emailenron/cnrankk2.log
python3 . -k 3 --dataset email_enron --mode rank --heur cn >> experiments/emailenron/cnrankk3.log
python3 . -k 5 --dataset email_enron --mode rank --heur cn >> experiments/emailenron/cnrankk5.log

for k in $(seq 7 29)
do
    python3 . -k $k --dataset email_enron --mode rank --heur cn >> experiments/emailenron/cnrankk$k.log
done
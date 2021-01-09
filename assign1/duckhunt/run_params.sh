#!/bin/bash
rm -f /tmp/player2server /tmp/server2player
mkfifo /tmp/player2server /tmp/server2player
rm ./log
for i in {99..85}; do
    export Prob=`bc -l <<< "$i/100"`
    for j in {1..5}; do
    echo "==================" >> log
    echo $Prob  >> log
    echo "==================" >> log
	./Skeleton server load ParadiseEmissions.in < /tmp/player2server | ./Skeleton verbose > /tmp/player2server 2>> log
    done
done
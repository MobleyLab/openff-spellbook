#!/bin/bash

date | tee flag.master.log 
time  for i in a b t i ; do  seq 0 200 | sed "s/^/$i/"
done  2>> flag.master.log | \
	xargs -P 4 -n 1 -I{} sh -c "mkdir $i{} ; cd $i{} ; python3 -u ../../../flag_measurements.py ../../db.pickle $i{} | tee flag.${i}{}.log " | \
		stdbuf -oL grep Query | tee -a flag.master.log
date | tee -a flag.master.log 

grep ' R ' */flag.*.log > RED.flags.out
grep ' Y ' */flag.*.log > YELLOW.flags.out

mkdir RED YELLOW 2> /dev/null

for p in a b t i ; do
	for n in $(seq 0 200); do
		redfile = "RED/RED.mol.smiles.${p}${n}.out"
		yellowfile = "YELLOW/YELLOW.mol.smiles.${p}${n}.out"
		sort -k17g ${p}${n}/mol*.smiles_with.${p}${n}.txt 2> /dev/null | tee \
			>(grep ' R ' > $redfile ) | \
			  grep ' Y ' > $yellowfile
		[ "$(head -n 1 $redfile)" ] || rm $redfile
		[ "$(head -n 1 $yellowfile)" ] || rm $yellowfile

	done
done

echo "Wrote RED.flags.out and YELLOW.flags.out"
echo "Wrote RED.mol.smiles.param.out YELLOW.mol.smiles.param.out"

echo "Use: grep RMS RED.flags.out | sort -k17gr' to sort"

# vim: ft=sh

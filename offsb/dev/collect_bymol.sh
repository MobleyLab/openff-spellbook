printf " %6s %6s %6s %s\n" count  MOLID QCID QCNAME
awk '{printf("%6d %6d %s\n", $2, $3, $4)}' $1  | uniq -c | sort -nr -k1

awk '{print $8 }' $1 | sort | uniq -c | sort      -k2.1d,2.2   -k2.3n,2.5

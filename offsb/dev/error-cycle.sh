while date ; do rm -f *.p *.xyz ; echo "Scanning" ; python -m offsb.ui.qca.errors  --datasets ../datasets  --save-xyz --report-out errors.log &> qca.log ; grep -A 2 Completes qca.log ; rm *nocat*.xyz *constr*.xyz ; date ; bash restart.{sh,py}  ; sleep $(( 60 * 10 )) ; done


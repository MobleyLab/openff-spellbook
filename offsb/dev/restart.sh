[ "${1}" ] || { "specify the restart.py file first." ; exit ; }
$1 OPT-RESTART.txt  task
$1 RESTART-TD-OPT.txt task
$1 TD-RESTART.txt service
# xargs -P 16 -a OPT-RESTART.txt -L 1 -I{} $1 {} task
# xargs -P 16 -a RESTART-TD-OPT.txt -L 1 -I{} $1 {} task
# xargs -P 16 -a TD-RESTART.txt -L 1 -I{} $1 {} service

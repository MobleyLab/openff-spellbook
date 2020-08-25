

import sys

with open(sys.argv[1], 'r') as fid:
    failed_runs = [int(x.strip('\n')) for x in fid.readlines()] 

total = int(sys.argv[2])

i=-1
j=-1
# want fig num -> xyz file
print("fig xyz")
while(j < total):
    if(j in failed_runs):
        j += 1
        continue
    else:
        i += 1
        j += 1
        print("{:3d} {:3d}".format(i,j))

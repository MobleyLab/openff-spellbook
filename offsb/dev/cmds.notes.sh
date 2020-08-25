

mkdir figs_and_xyz

# get the indices of completed sets
cp index_dataset.txt index_complete.txt
for i in $(grep -B 5 nothing measure_parameters.out | grep Result | awk '{print $2 }') ; do
	sed -i "/\<${i}\>/d" index_complete.txt
done

# copy the completed xyz into a folder
for i in $(cut -d' ' -f 1 index_complete.txt ) ; do
	echo xyz/mol_${i}.min.xyz
done | xargs -n 1 -I{} cp {} figs_and_xyz/

#view the xyz
vmd -m $(for i in $(cut -d' ' -f 1 ../index_complete.txt ) ; do echo mol_${i}.min.xyz ; done )

import pickle
import offsb.qcarchive.qcatree as qca

with open("QCA.p", 'rb') as fid:
    QCA = pickle.load(fid)
with open("QCA.db.p", 'rb') as fid:
    QCA.db = pickle.load(fid).db

nodes = QCA.combine_by_entry(fn=qca.match_canonical_isomeric_explicit_hydrogen_smiles)

print( "There are", len(nodes), "unique mols")

#print( [n.children for n in nodes])

for n in nodes:
    entry = QCA.db[QCA.node_index[n.children[0]].payload]['entry']
    print( len(n.children), entry.name)
    for c in n.children:
        child_entry = QCA.db[QCA.node_index[ c].payload]['entry']
        attr = child_entry.attributes['canonical_isomeric_explicit_hydrogen_smiles']
        print("    ", QCA.node_index[ c].name, QCA.node_index[ c].payload, attr)


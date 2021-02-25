#!/usr/bin/env python3 
import qcfractal.interface as ptl
import offsb.qcarchive.qcatree as qca
import treedi.node as Node


def generate_status_report( collection, name, logname):
    """
    Iterates a dataset, checking for errors, and saves errors to the log file.
    """

    client = ptl.FractalClient()
    QCA = qca.QCATree( "QCA", root_payload=client, node_index=dict(), db=dict())
    for s in sets:
        ds = client.get_collection( collection, name)
        QCA.build_index( ds, drop=["Gradient"], keep_specs=["default"],
            start=0, limit=0)

    default = [n for n in QCA.node_iter_depth_first(QCA.root()) 
        if "QCS-default" in n.payload]
    mindepth = QCA.node_depth(default[0])
    tdfailures= 0

    fid = open(logname, 'w')

    for node in QCA.node_iter_dive(default):
        i = QCA.node_depth(node) - mindepth+1
        status=""
        statbar= "    "
        errmsg = ""
        hasstat = False
        qcp = QCA.db[node.payload]['data']
        try:
            hasstat = "status" in QCA.db[node.payload]["data"]
        except Exception:
            pass
        if hasstat:
            status = qcp["status"][:]
            if status != "COMPLETE" and node.name == "Optimization":
                try:
                    statbar = "----"
                    client = qcp['client']
                    print("error", qcp['error'], qcp['error'] is None)
                    if not (qcp['error'] is None):
                        key=int(qcp['error'])
                        err = list(client.query_kvstore(key).values())[0]
                        errmsg += "####ERROR####\n"
                        errmsg += "Error type: " + err['error_type'] + "\n"
                        errmsg += "Error:\n" + err['error_message']
                        errmsg += "############\n"
                    elif status == "ERROR":
                        errmsg += "####FATAL####\n"
                        errmsg += "Job errored and no error on record. "
                        errmsg += "Input likely bogus; check your settings\n"
                        errmsg += "############\n"
                    if not (qcp['stdout'] is None):
                        key=int(qcp['stdout'])
                        msg = list(client.query_kvstore(key)).values())[0]
                        errmsg += "----ISSUE----\n"
                        errmsg += "Status was not complete; stdout is:\n"
                        errmsg += msg
                        errmsg += "-------------\n"
                    if not (qcp['stderr'] is None):
                        key=int(qcp['stderr'])
                        msg = list(client.query_kvstore(key)).values())[0]
                        errmsg += "####ISSUE####\n"
                        errmsg += "Status was not complete; stderr is:\n"
                        errmsg += msg
                        errmsg += "-------------\n"
                except Exception as e:
                    fid.write("Internal issue:\n" + str(e) + '\n')

            if status != "COMPLETE" and node.name == "TorsionDrive":
                statbar = "XXXX"
        if node.name == "TorsionDrive" and status != "COMPLETE":
            tdfailures += 1
        fid.write("{:2d} {} {} {}\n".format(tdi,statbar*i,
            " ".join([str(node.index),str(node.name), str(node.payload)]),
            status))
        if errmsg != "":
            fid.write("\n{}\n".format(errmsg))

    fid.close()

    return QCA

if __name___ == "__main__":
    QCA = generate_status_report( 
            "TorsionDriveDataset", "OpenFF Substituted Phenyl Set 1",
            logname="TorsionDriveDataset.OpenFF-Substituted-Phenyl-Set-1.log")


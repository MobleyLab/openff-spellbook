#!/usr/bin/env python

import qcfractal.interface as ptl

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('server_uri', nargs='?', default=None, help='The QCArchive server to connect to')
    parser.add_argument('--all-specs', action="store_true", help='Include the specs in the list')
    parser.add_argument('--only-specs', nargs='+', default=None, help='Only include these specs in the output (datasets without this spec will not be listed)')
    parser.add_argument('--only-types', nargs='+', default=None, help='Only include these dataset types')

    args = parser.parse_args()
    server_uri = args.server_uri

    client = None
    if server_uri is None:
        client = ptl.FractalClient()
    elif server_uri == "from_file":
        client = ptl.FractalClient.from_file()
    else:
        client = ptl.FractalClient(server_uri, verify=False)

    args.only_types = [x.lower() for x in args.only_types] if args.only_types else args.only_types
    for row in client.list_collections().iterrows():
        if args.only_types is not None and row[0][0].lower() not in args.only_types:
            continue
        out_str = ""
        if server_uri is not None:
            out_str = str(client.address) + " / "
        out_str += row[0][0] + " " + row[0][1] + " "
        do_print = True
        if args.all_specs or args.only_specs:
            ds = client.get_collection(row[0][0], row[0][1])
            if hasattr(ds, "list_specifications"):
                specs = list(ds.list_specifications().index)
                if args.only_specs:
                    specs = [s for s in specs if s in args.only_specs]

                out_str += "/ " + " ".join(specs)
            else:
                do_print = False
        if do_print:
            print(out_str)

if __name__ == "__main__":
    main()

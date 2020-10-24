#!/usr/bin/env python

import qcfractal.interface as ptl

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('server_uri', nargs='?', default=None, help='The QCArchive server to connect to')
    parser.add_argument('--all-specs', action="store_true", help='Include the specs in the list')
    parser.add_argument('--only-specs', nargs='+', default=None, help='Only include these specs in the output (datasets without this spec will not be listed)')

    args = parser.parse_args()
    server_uri = args.server_uri

    client = None
    if server_uri is None:
        client = ptl.FractalClient()
    elif server_uri == "from_file":
        client = ptl.FractalClient.from_file()
    else:
        client = ptl.FractalClient(server_uri, verify=False)

    for row in client.list_collections().iterrows():
        if server_uri is not None:
            print(client.address, "/", end=" ")
        print(row[0][0], row[0][1], end=" ")
        if args.all_specs or args.only_specs:
            ds = client.get_collection(row[0][0], row[0][1])
            if hasattr(ds, "list_specifications"):
                specs = list(ds.list_specifications().index)
                if args.only_specs:
                    specs = [s for s in specs if s in args.only_specs]
                print("/", " ".join(specs), end=" ")
        print()


if __name__ == "__main__":
    main()

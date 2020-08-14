#!/usr/bin/env python

import qcportal as ptl

def main():
    client = ptl.FractalClient()
    for row in client.list_collections().iterrows():
        print(row[0][0], row[0][1])

if __name__ == "__main__":
    main()

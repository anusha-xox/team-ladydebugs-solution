#!/usr/bin/env python3
import csv, re, sys, os

INFILE  = "merged_12_months_hmis.csv"      # input file
OUTFILE = "merged_12_months_hmis_clean.csv"  # output file

allowed = re.compile(r'[^A-Za-z0-9_]+')

with open(INFILE, "r", newline="", encoding="utf-8") as fin, \
     open(OUTFILE, "w", newline="", encoding="utf-8") as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)

    header = next(reader)
    new_header = []
    mapping = {}
    for col in header:
        # BigQuery col rules: start with letter/underscore; only letters, numbers, underscores
        cleaned = allowed.sub("_", col).strip("_")
        if not cleaned or not re.match(r'^[A-Za-z_]', cleaned):
            cleaned = f"col_{cleaned or 'unnamed'}"
        # avoid duplicate names by appending an index if needed
        base = cleaned
        i = 1
        while cleaned in new_header:
            i += 1
            cleaned = f"{base}_{i}"
        mapping[col] = cleaned
        new_header.append(cleaned)

    # show mapping
    print("Header mapping (original -> cleaned):")
    for k, v in mapping.items():
        if k != v:
            print(f" - {k} -> {v}")

    writer.writerow(new_header)
    for row in reader:
        writer.writerow(row)

print(f"✅ Wrote: {OUTFILE} ({len(new_header)} columns)")


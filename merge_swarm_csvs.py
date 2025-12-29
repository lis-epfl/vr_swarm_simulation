#!/usr/bin/env python3
"""
Merge multiple swarm position CSV files into a single file.
"""

import os
import csv
from pathlib import Path

# Input directory with CSV files
csv_dir = r"D:\advaith\unity-run-files\swarm_CSV"
output_file = os.path.join(csv_dir, "swarm_positions_merged.csv")

# Find all swarm_positions CSV files
csv_files = sorted([f for f in os.listdir(csv_dir) if f.startswith("swarm_positions_") and f.endswith(".csv") and f != "swarm_positions_merged.csv"])

if not csv_files:
    print("No swarm position CSV files found!")
    exit()

print(f"Found {len(csv_files)} CSV files to merge:")
for f in csv_files:
    print(f"  - {f}")

# Read all data
all_rows = []
header = None

for csv_file in csv_files:
    filepath = os.path.join(csv_dir, csv_file)
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            if len(rows) > 0:
                # Save header from first file
                if header is None:
                    header = rows[0]
                
                # Add data rows (skip header)
                if len(rows) > 1:
                    all_rows.extend(rows[1:])
                    print(f"  {csv_file}: {len(rows)-1} data rows")
    except Exception as e:
        print(f"  Error reading {csv_file}: {e}")

# Sort by capture_time (first column)
all_rows.sort(key=lambda x: x[0] if len(x) > 0 else "")

# Write merged CSV
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    
    # Write header
    if header:
        writer.writerow(header)
    
    # Write all data
    writer.writerows(all_rows)

print(f"\n✓ Merged {len(all_rows)} total rows into: {output_file}")
print(f"  Columns: {header}")

import csv
import collections

# Load data
solute_solvent_map = collections.defaultdict(set)
rows = 0

with open('raw/ori_data.csv', 'r', errors='ignore') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows += 1
        smi = row['SMILES']
        solvent = row['Solvent']
        if smi and solvent:
            solute_solvent_map[smi].add(solvent)

total_rows = rows
unique_solutes = len(solute_solvent_map)
all_solvents = set()
for solv_set in solute_solvent_map.values():
    all_solvents.update(solv_set)
unique_solvents = len(all_solvents)

# Distribution
counts_per_solute = [len(solvents) for solvents in solute_solvent_map.values()]
dist_counts = collections.Counter(counts_per_solute)

# Print Report
print(f"Total Data Points: {total_rows}")
print(f"Total Unique Solutes: {unique_solutes}")
print(f"Total Unique Solvents: {unique_solvents}")
print(f"Average Data Points per Solute: {total_rows / unique_solutes:.2f}")

print("\n--- Distribution: How many solvents does a solute have? ---")
sorted_counts = sorted(dist_counts.items())
# Show first 10 counts and then aggregate the tail if long
for num_solvents, count in sorted_counts:
    percentage = (count / unique_solutes) * 100
    print(f"{num_solvents} Solvents: {count} solutes ({percentage:.1f}%)")
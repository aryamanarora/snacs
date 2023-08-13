import csv

files = ['v0.1.csv', 'v0.2.csv']

for file in files:
    with open(f"raw_data/{file}", 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

        with open(f"../data/{file}", 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for row in data[2:]:
                for d in [13, 14]:
                    if not row[d].startswith('p.') and row[d] not in ['_', '']:
                        row[d] = 'p.' + row[d]
                writer.writerow(row)
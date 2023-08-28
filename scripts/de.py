import csv

files = {'v0.1.csv': 'de-lp.conllulex', 'v0.2.csv': 'de-lp-new.conllulex'}

for file in files:
    print(file)

    # read data
    data = None
    with open(f"raw_data/{file}", 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    # write in proper conllulex format
    with open(f"../data/{files[file]}", 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for row in data[2:]:

            # ensure _ for empty
            row = [x.strip() for x in row]
            if not row[0].startswith('#') and row[0] != '':
                row = ['_' if x == '' else x for x in row]

            # fix multi-token
            if row[0].endswith('-1'):
                full_word = row[::]
                full_word[0] = row[0].split('-')[0]
                full_word[1] = full_word[2]
                print(full_word)
                writer.writerow(full_word)
            
            # ignore dashed (if not metadata)
            if '-' in row[0] and not row[0].startswith('#'): continue

            # fix snacs labels
            for d in [13, 14]:
                if not row[d].startswith('p.') and row[d] not in ['_', '']:
                    row[d] = 'p.' + row[d]

            # write row
            writer.writerow(row)
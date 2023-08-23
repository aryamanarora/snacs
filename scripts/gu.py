import stanza
import csv

# nlp = stanza.Pipeline(lang='hi', processors='tokenize,pos,lemma,depparse', tokenize_pretokenized=True)

# columns
# id, word, lemma, upos, xpos, feats, head, deprel, [deps, misc], smwe, [lexcat, lexlemma], ss, ss2, [wmwe, wlemma, lextag]

res = ''
for i in range(1, 28):
    print(i)
    with open(f'raw_data/gu/NR_Ch{i}.csv', 'r') as fin:
        reader = csv.reader(fin)
        next(reader)

        # get all sentences from this file,
        # maintaining our tokenisation
        sents, sent = [], []
        for row in reader:
            if row[1] == '':
                if sent != []: sents.append(sent)
                sent = []
            else:
                sent.append(row)
        sents.append(sent)

        # merge sentences for stanza
        # flat = '\n'.join([' '.join([y[1] for y in x]) for x in sents])
        # print(flat)
        # input()
        # doc = nlp(flat)
        # print(doc.sentences[0])
        # input()

        # to conllulex
        for sent_id, sent in enumerate(sents):
            # print(sent.text, [x[0] for x in sents[sent_id]])
            res += f'\n# sent_id = lp_gu_{i}_{sent_id}'
            sent = [x[1] for x in sent]
            res += f'\n# text = {" ".join(sent)}'
            for word_id, word in enumerate(sent):
                # generate row for each word, first normal CONLLU stuff
                # output = [word_id, word, word.lemma, word.upos, word.xpos, word.feats, word.head, word.deprel]
                output = [word_id, word]
                output.extend([''] * 8)

                non_initial = False
                if sents[sent_id][word_id][6]:
                    if sents[sent_id][word_id][6].split(':')[1] != '1':
                        non_initial = True

                # now LEX features
                output.extend([sents[sent_id][word_id][6], 'P' if sents[sent_id][word_id][2] else '', sents[sent_id][word_id][3] if not non_initial else ''])
                output.append('p.' + sents[sent_id][word_id][4])
                output.append('p.' + sents[sent_id][word_id][5])
                output.extend([''] * 4)
                output = ['_' if x in ['', 'p.'] else x for x in output]

                # add to output
                res += '\n' + '\t'.join(map(str, output))
            res += '\n'

with open('../data/gu-lp.conllulex', 'w') as fout:
    fout.write(res)
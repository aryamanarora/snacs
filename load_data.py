import conllu
from transformers import AutoTokenizer
import glob
from tqdm import tqdm

conllulex = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head','deprel', 'deps',
             'misc', 'smwe', 'lexcat', 'lexlemma', 'ss', 'ss2', 'wmwe', 'wcat', 'wlemma', 'lextag']

def tokenize_and_align(
    file: str,
    tokenizer: AutoTokenizer,
    add_space: bool=True,
    hide_non_adp: bool=True,
    o_to_b: bool=True):
    
    print(file)
    res = []
    with open(file, 'r') as fin:

        failed = 0
        for sent in tqdm(conllu.parse_incr(fin, fields=conllulex)):
            text = sent.metadata['text']

            tokens = []
            mask = []
            labels = []
            work = True

            smwe_tags = {}

            for i, token in enumerate(sent):
                if not isinstance(token['id'], int): continue

                if token['smwe'] != '_':
                    smwe, pos = token['smwe'].split(':')
                    if smwe not in smwe_tags:
                        if token['ss'] != '_':
                            smwe_tags[smwe] = token['ss'] + '-' + token['ss2']
                            token['lextag'] = 'B-' + smwe_tags[smwe]
                        else:
                            token['lextag'] = 'O'
                    else:
                        token['lextag'] = 'I-' + smwe_tags[smwe]
                elif token['ss'] != '_':
                    token['lextag'] = 'B-' + token['ss'] + '-' + token['ss2']
                else:
                    token['lextag'] = 'O'

                tok = None
                if i == 0 or not add_space: tok = tokenizer.encode(token['form'])
                else: tok = tokenizer.encode(' ' + token['form'])

                labels.append(token['lextag'])
                labels.extend(['None' for _ in range(len(tok) - 1)])
                mask.append(1)
                mask.extend([0 for _ in range(len(tok) - 1)])
                tokens.extend(tok)
            
            if work:
                res.append([tokens, mask, labels])

    for i in res:
        for j in range(len(i[0])):
            print(f"{tokenizer.decode(i[0][j]):<15} {i[1][j]:<10} {i[2][j]:<30}")
        input()
    if failed != 0:
        print(f"Failed to parse {failed} sentences, did {len(res)}.")
    else:
        print(f"Done: {len(res)} sentences.")

    return res


def main():
    for file in glob.glob("data/*.conllulex"):
        print(file)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenize_and_align(file, tokenizer)

if __name__ == "__main__":
    main()
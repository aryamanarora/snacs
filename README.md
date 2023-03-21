# snacs

Models for parsing SNACS datasets. See [Schneider et al. (2018)](https://aclanthology.org/P18-1018/).

## Results

I'm sure it's possible to get competitive results with [Liu et al. (2021)](https://aclanthology.org/2021.mwe-1.6/) using very little additional stuff on top of the language model.

| Model | Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| `bert-base-cased` + Linear | `en-streusle` | 63.3 | 65.2 | 64.3 |
| `roberta-base` + Linear | `en-streusle` | 66.6 | 70.6 | 68.5 |
| Liu et al. (2021) | `en-streusle` | | | **70.9** |
| Schneider et al. (2018) | `en-streusle` | | | 55.7 |
| `bert-base-cased` + Linear | `en-lp` | 41.7 | 41.9 | 41.8 |
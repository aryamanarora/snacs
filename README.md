# snacs

Models for parsing SNACS datasets. See [Schneider et al. (2018)](https://aclanthology.org/P18-1018/).

## Results

| Model | W&B | Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| `bert-base-cased` + Linear | `dauntless-sun-13` | `en-streusle` | 63.3 | 65.2 | 64.3 |
| `roberta-base` + Linear | `feasible-dragon-20` | `en-streusle` | 77.0 | 79.6 | **78.2** |
| Liu et al. (2021) | | `en-streusle` | | | 70.9 |
| Schneider et al. (2018) | | `en-streusle` | | | 55.7 |
| `bert-base-cased` + Linear | `neat-wood-17` | `en-lp` | 41.7 | 41.9 | 41.8 |
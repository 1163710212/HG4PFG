Code for  Paper[**“Progressive Preference Guidance for Item-Side Fairness in Interactive Recommendation”**]


## Installation

```bash
torch==2.5.1
numpy==2.2.5
pandas==2.3.0
tqdm==4.67.1
```

## Examples to run the code

- #### KuaiRand

Train user mode:

```
bash scripts/run_multibehavior.sh
```

Then train agent model:

```
bash scripts/train_hrl4pfg.sh
```

## Credit

This repo is based on [KuaiSim]

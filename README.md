# NER Project for Single Bell
## Improve Named Entity Recognition with Entity-wise Relationship
### XING JInbo, LU Fanbin, ZHANG Yuechen
#### For FULL version of the code, please refer to [this site](https://github.com/JulianJuaner/CSCI5640_NLP).
#### This branch only includes main results in the report. For full experiments, please refer to other branches (julian/jinbo) of this repository. Feel free to tell us if you have any questions.
***
# Data Preprocessing
You can download CoNLL03 dataset from huggingface, or the official website. If you are using the huggingface format, you can convert the data to corpus by:
```
python hf_data.py
```
To enable our data augmentation, you need to generate the dictionary (train_dict.json) first.
```
python corpus_ner_dict.py
```
Then you need to move it into its flair cache locations.
***
# Training
For following training scripts, please use the flair in this repository because we modified it in the expreriments. You can global search keyword 'MODIFIED' to see what we have done in the original codebase.

```
export PYTHONPATH=/.../CSCI5640_NLP/flair:$PYTHONPATH
```
you will show a print message to indicate whether you are using the correct version after importing the flair package.
## Train with Different Embeddings
```
python exps/embedding/embedding_ELMo_different_configs.py
python exps/embedding/embedding_concat_and_decoding.py
python exps/embedding/embedding_different_pretrainmodels.py
```
## Train with Penalty Loss
```
python exps/penalty_loss/trainval.py
```
## Train with Entity Shuffling
```
python exps/entity_shuffle/trainval.py
```
## Train with Shuffling & Entity Relationship
```
python exps/entity_shuffle_ER/trainval.py
```
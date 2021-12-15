from datasets import load_dataset
import json
from tqdm import tqdm

def json_save():
    dataset = load_dataset("conll2003")
    print(dataset)
    new_d = dict()
    new_d['train'] = []
    new_d['validation'] = []
    new_d['test'] = []
    for item in dataset['train']:
        # print(item)
        new_d['train'] += [item]
    for item in dataset['validation']:
        new_d['validation'] += [item]
    for item in dataset['test']:
        new_d['test'] += [item]

    with open("hf_dataset.json", 'w') as o:
        json.dump(new_d, o)

def json_load():
    d = dict()
    names=["O","B-PER","I-PER","B-ORG",
            "I-ORG","B-LOC","I-LOC","B-MISC","I-MISC"]
    with open("hf_dataset.json", 'r') as f:
        d = json.load(f)
    padding = " NNP I-NP "
    with open("eng.train", 'w') as f:
        for item in tqdm(d['train']):
            for id, token in enumerate(item['tokens']):
                new_line = token + padding + names[item['ner_tags'][id]] + '\n'
                f.write(new_line)
            f.write("\n")
    with open("eng.testa", 'w') as f:
        for item in tqdm(d['validation']):
            for id, token in enumerate(item['tokens']):
                new_line = token + padding + names[item['ner_tags'][id]] + '\n'
                f.write(new_line)
            f.write("\n")
    with open("eng.testb", 'w') as f:
        for item in tqdm(d['test']):
            for id, token in enumerate(item['tokens']):
                new_line = token + padding + names[item['ner_tags'][id]] + '\n'
                f.write(new_line)
            f.write("\n")
json_save()
json_load()
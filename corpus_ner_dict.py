import json
import os
from tqdm import tqdm

d = dict()
d["PER"] = set()
d["ORG"] = set()
d["LOC"] = set()
d["MISC"] = set()

prev_attr = "O"
prev_word = ""
count = 0
with open("./eng.train") as f:
    for i, each in tqdm(enumerate(f.readlines())):
        line  = each.strip()
        word_attr = line.split(" ")
        if len(word_attr) == 4 and word_attr[-1] != "O":
            if word_attr[-1].split('-')[0] != "I":
                if prev_word != "":
                    d[prev_attr].update([prev_word])
                    prev_attr = "O"
                    prev_word = ""
                count += 1

            word = word_attr[0]
            attr = word_attr[-1].split("-")[-1]
        
            if prev_attr == attr:
                if prev_word != "":
                    prev_word = prev_word + " " + word
                else:
                    prev_word = word
            else:
                prev_word = word
            prev_attr = attr
        else:
            if prev_word != "":
                # print(prev_word)
                d[prev_attr].update([prev_word])
                # count += 1
            prev_attr = "O"
            prev_word = ""

d["PER"] = list(d["PER"])
d["ORG"] = list(d["ORG"])
d["LOC"] = list(d["LOC"])
d["MISC"] = list(d["MISC"])
print(count, len(d["PER"]) + len(d["ORG"]) + len(d["LOC"]) +len(d["MISC"]))
# print(d["MISC"])
print(1653 + 1657 + 1612 + 671)
with open("train_dict.json", "w") as o:
    json.dump(d, o)

# print(d)
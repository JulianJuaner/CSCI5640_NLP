'''
The Chinese University of Hong Kong
CSCI5640 Natural Language Processing
Course Project
Group:             Single Bell
Members:           LU, Fanbin and XING, Jinbo and ZHANG, Yuechen
PIC for this file: ZHANG Yuechen (zhangyc@link.cuhk.edu.hk)
'''

from conll03_shuf import CONLL_03
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from sequence_tagger_model import SequenceTagger
from trainer import ModelTrainer
from flair.visual.training_curves import Plotter

# 1. get the corpus
corpus = CONLL_03()
print(corpus)

# 2. what label do we want to predict?
label_type = 'ner'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# 4. initialize embedding stack with Flair and GloVe
embedding_types = [
    WordEmbeddings('glove'),
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]

embeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type,
                        use_crf=True)

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/entity_shuf_er',
              learning_rate=0.2,
              mini_batch_size=32,
              write_weights=True,
              monitor_test=True,
              patience=5,
              max_epochs=150)

plotter = Plotter()
plotter.plot_training_curves('resources/taggers/entity_shuf_er/loss.tsv')
plotter.plot_weights('resources/taggers/entity_shuf_er/weights.txt')
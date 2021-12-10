# from flair.datasets import CONLL_03
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
from conll03_shuf import CONLL_03_Shuf

# 1. get the corpus
corpus = CONLL_03_Shuf("/research/d4/gds/yczhang21/project/CSCI5640_NLP")
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
trainer.train('exps/e07-15',
              learning_rate=0.1,
              mini_batch_size=32,
              write_weights=True,
              monitor_test=True,
              max_epochs=150)

plotter = Plotter()
plotter.plot_training_curves('exps/e07-15/loss.tsv')
plotter.plot_weights('exps/e07-15/weights.txt')
# from flair.datasets import CONLL_03
from flair.datasets import CONLL_03
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from trainer import ModelTrainer
from flair.visual.training_curves import Plotter
from torch.optim.lr_scheduler import OneCycleLR
import torch

from conll03_shuf import CONLL_03_Shuf

# 1. get the corpus
corpus = CONLL_03_Shuf("/research/d4/gds/yczhang21/project/CSCI5640_NLP")
print(corpus)

# 2. what label do we want to predict?
label_type = 'ner'

# 3. make the label dictionary from the corpus
tag_dictionary = corpus.make_label_dictionary(label_type=label_type)
print(tag_dictionary)

# 4. initialize embedding stack with Flair and GloVe
embeddings = TransformerWordEmbeddings(
    model='roberta-large',
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
)

# 5. initialize sequence tagger
tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type='ner',
    use_crf=True,
    reproject_embeddings=True,
)

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('exps/e09-dataaug-large',
              learning_rate=5.0e-6,
              mini_batch_size=4,
              mini_batch_chunk_size=1,
              max_epochs=20,
              scheduler=OneCycleLR,
              embeddings_storage_mode='none',
              weight_decay=0.,
              monitor_test=True,
              optimizer=torch.optim.AdamW
)

plotter = Plotter()
plotter.plot_training_curves('exps/e09-dataaug-large/loss.tsv')
plotter.plot_weights('exps/e09-dataaug-large/weights.txt')
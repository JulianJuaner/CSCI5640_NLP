'''
The Chinese University of Hong Kong
CSCI5640 Natural Language Processing
Course Project
Group:             Single Bell
Members:           LU, Fanbin and XING, Jinbo and ZHANG, Yuechen
PIC for this file: XING, Jinbo (JinboXING@link.cuhk.edu.hk)
'''
from flair.data import Corpus
from flair.datasets import CONLL_03
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings

from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.optim.lr_scheduler import OneCycleLR

corpus: Corpus = CONLL_03('../')
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

PretrainedEmbeddings = 'Flair'
if PretrainedEmbeddings == 'Flair':
    embeddings = StackedEmbeddings([
        WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ])
elif PretrainedEmbeddings == 'BERT':
    embeddings = TransformerWordEmbeddings(
    model='bert-base-uncased',
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
)
elif PretrainedEmbeddings == 'XLNet':
    embeddings = TransformerWordEmbeddings(
    model='xlnet-base-cased',
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
)
elif PretrainedEmbeddings == 'RoBERTa':
    embeddings = TransformerWordEmbeddings(
    model='roberta-large',
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
)
elif PretrainedEmbeddings == 'XLM-R':
    embeddings = TransformerWordEmbeddings(
    model='xlm-roberta-large',
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
)



tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type='ner',
    use_crf=False,
    use_rnn=False,
    reproject_embeddings=False,
)

trainer = ModelTrainer(tagger, corpus)



if PretrainedEmbeddings == 'Flair':
    trainer.train('resources/taggers/{}'.format(PretrainedEmbeddings),
                train_with_dev=False,
                max_epochs=150)
else:
    trainer.train('resources/taggers/{}'.format(PretrainedEmbeddings),
              learning_rate=5.0e-6,
              mini_batch_size=4,
              mini_batch_chunk_size=1,
              max_epochs=20,
              scheduler=OneCycleLR,
              embeddings_storage_mode='none',
              weight_decay=0.,
              optimizer=torch.optim.AdamW
              )
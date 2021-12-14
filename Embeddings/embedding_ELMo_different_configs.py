'''
The Chinese University of Hong Kong
CSCI5640 Natural Language Processing
Course Project
Group:             Single Bell
Members:           LU, Fanbin and XING, Jinbo and ZHANG, Yuechen
PIC for this file: XING, Jinbo (JinboXING@link.cuhk.edu.hk)
'''

import torch
from flair.datasets import CONLL_03
corpus = CONLL_03('../')
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# you need to install AllenNLP==0.9
from flair.embeddings import ELMoEmbeddings

from flair.models import SequenceTagger
from torch.optim.lr_scheduler import OneCycleLR
from flair.trainers import ModelTrainer

for ELMO_MODE in ['small', 'medium', 'original']:
    embeddings = ELMoEmbeddings(
        model=ELMO_MODE,
        embedding_mode='all'
    )

    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type='ner',
        use_crf=True,
        use_rnn=True,
        rnn_layers=2,
        rnn_type='LSTM',
        reproject_embeddings=False,
    )

    trainer = ModelTrainer(tagger, corpus)
    trainer.train('resources/taggers/elmo-{}'.format(ELMO_MODE),
                learning_rate=0.001,
                mini_batch_size=4,
                mini_batch_chunk_size=1,
                max_epochs=20,
                scheduler=OneCycleLR,
                embeddings_storage_mode='none',
                weight_decay=0.,
                optimizer=torch.optim.Adam
                )


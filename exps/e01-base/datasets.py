import flair
from pathlib import Path
from typing import Union
from flair.datasets.sequence_labeling import ColumnCorpus
import logging
log = logging.getLogger("flair")

class CONLL_03_shuf(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
            **corpusargs,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put the eng.testa, .testb, .train
        files in a folder called 'conll_03'. Then set the base_path parameter in the constructor to the path to the
        parent directory where the conll_03 folder resides.
        If using entity linking, the conll03 dateset is reduced by about 20 Documents, which are not part of the yago dataset.
        :param base_path: Path to the CoNLL-03 corpus (i.e. 'conll_03' folder) on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "np", 3: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'WARNING: CoNLL-03 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://www.clips.uantwerpen.be/conll2003/ner/"'
            )
            log.warning("-" * 100)

        super(CONLL_03_shuf, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            **corpusargs,
        )
import logging
import flair
import random
import json
from flair.data import Corpus, MultiCorpus, FlairDataset, Sentence, Token
from flair.datasets.sequence_labeling import ColumnDataset
from flair.datasets.base import find_train_dev_test_files
from torch.utils.data import ConcatDataset
from typing import Union, Dict, List, Optional
from pathlib import Path
from flair.datasets import DataLoader


log = logging.getLogger("flair")

class MultiFileColumnCorpus(Corpus):
    def __init__(
            self,
            column_format: Dict[int, str],
            train_files=None,
            test_files=None,
            dev_files=None,
            tag_to_bioes=None,
            column_delimiter: str = r"\s+",
            comment_symbol: str = None,
            encoding: str = "utf-8",
            document_separator_token: str = None,
            skip_first_line: bool = False,
            in_memory: bool = False,
            label_name_map: Dict[str, str] = None,
            banned_sentences: List[str] = None,
            **corpusargs,
    ):
        # get train data
        train = ConcatDataset([
            ColumnDataset_Shuf(
                train_file,
                column_format,
                tag_to_bioes,
                encoding=encoding,
                comment_symbol=comment_symbol,
                column_delimiter=column_delimiter,
                banned_sentences=banned_sentences,
                in_memory=in_memory,
                document_separator_token=document_separator_token,
                skip_first_line=skip_first_line,
                label_name_map=label_name_map,
            ) for train_file in train_files
        ]) if train_files and train_files[0] else None

        # read in test file if exists
        test = ConcatDataset([
            ColumnDataset(
                test_file,
                column_format,
                tag_to_bioes,
                encoding=encoding,
                comment_symbol=comment_symbol,
                column_delimiter=column_delimiter,
                banned_sentences=banned_sentences,
                in_memory=in_memory,
                document_separator_token=document_separator_token,
                skip_first_line=skip_first_line,
                label_name_map=label_name_map,
            ) for test_file in test_files
        ]) if test_files and test_files[0] else None

        # read in dev file if exists
        dev = ConcatDataset([
            ColumnDataset(
                dev_file,
                column_format,
                tag_to_bioes,
                encoding=encoding,
                comment_symbol=comment_symbol,
                column_delimiter=column_delimiter,
                banned_sentences=banned_sentences,
                in_memory=in_memory,
                document_separator_token=document_separator_token,
                skip_first_line=skip_first_line,
                label_name_map=label_name_map,
            ) for dev_file in dev_files
        ]) if dev_files and dev_files[0] else None

        super(MultiFileColumnCorpus, self).__init__(train, dev, test, **corpusargs)

class ColumnCorpus(MultiFileColumnCorpus):
    def __init__(
            self,
            data_folder: Union[str, Path],
            column_format: Dict[int, str],
            train_file=None,
            test_file=None,
            dev_file=None,
            autofind_splits: bool = True,
            name: Optional[str] = None,
            **corpusargs,
    ):
        # find train, dev and test files if not specified
        dev_file, test_file, train_file = \
            find_train_dev_test_files(data_folder, dev_file, test_file, train_file, autofind_splits)
        super(ColumnCorpus, self).__init__(
            column_format,
            dev_files=[dev_file] if dev_file else [],
            train_files=[train_file] if train_file else [],
            test_files=[test_file] if test_file else [],
            name=name if data_folder is None else str(data_folder),
            **corpusargs
        )

class ColumnDataset_Shuf(ColumnDataset):
    # special key for space after
    SPACE_AFTER_KEY = "space-after"

    def __init__(
            self,
            path_to_column_file: Union[str, Path],
            column_name_map: Dict[int, str],
            tag_to_bioes: str = None,
            column_delimiter: str = r"\s+",
            comment_symbol: str = None,
            banned_sentences: List[str] = None,
            in_memory: bool = False,
            document_separator_token: str = None,
            encoding: str = "utf-8",
            skip_first_line: bool = False,
            label_name_map: Dict[str, str] = None,
            shuf_percentage: float = 0.25,
    ):
        super(ColumnDataset_Shuf, self).__init__(
            path_to_column_file, column_name_map, tag_to_bioes, column_delimiter, comment_symbol, banned_sentences,
            in_memory, document_separator_token, encoding, skip_first_line, label_name_map
        )
        with open("/research/d4/gds/yczhang21/project/CSCI5640_NLP/train_dict.json", "r") as f:
            self.dictionary = json.load(f)
        # print(self.dictionary)
        self.shuf_percentage = shuf_percentage


    def __getitem__(self, index: int = 0) -> Sentence:

        # if in memory, retrieve parsed sentence
        if self.in_memory:
            sentence = self.sentences[index]
    
        # else skip to position in file where sentence begins
        else:
            with open(str(self.path_to_column_file), encoding=self.encoding) as file:
                file.seek(self.indices[index])
                
                # DONE: shuffle sentence.
                line = self._read_next_sentence(file)
                new_line = []
                prev_words = []
                prev_attr = "O"
                padding_attr = " NNP I-NP "

                for idx, word_line in enumerate(line):
                    word_attr = word_line.strip().split(" ")
                    if len(word_attr) == 4 and word_attr[-1] != "O":
                        word = word_attr[0]
                        attr = word_attr[-1].split("-")[-1]
                    
                        if prev_attr == attr:
                            if prev_words != []:
                                prev_words += [word_line]
                            else:
                                prev_words += [word_line]
                        else:
                            prev_words += [word_line]
                        prev_attr = attr

                    else:
                        # if the word is not valid, release previous words.
                        if prev_words != []:
                            # still having probability to be not changed (else part)
                            if random.random() < self.shuf_percentage:
                                new_words = random.sample(self.dictionary[prev_attr], 1)[0]
                                new_words = new_words.split(" ")
                                # log.info(new_words)
                                for new_word in new_words:
                                    new_word = new_word + padding_attr + "I-" + prev_attr + "\n"
                                    new_line += [new_word]
                            else:
                                for prev_word in prev_words:
                                    new_line += [prev_word]

                        new_line += [word_line]
                        prev_attr = "O"
                        prev_words = []

                # at last, if there exist a prev_word, do same thing.
                if prev_words != []:
                    # still having probability to be not changed (else part)
                    if random.random() < self.shuf_percentage:
                        new_words = random.sample(self.dictionary[prev_attr], 1)[0]
                        new_words = new_words.split(" ")
                        # log.info(new_words)
                        for new_word in new_words:
                            new_word = new_word + padding_attr + "I-" + prev_attr + "\n"
                            new_line += [new_word]
                    else:
                        for prev_word in prev_words:
                            new_line += [prev_word]

                    # visualization
                    # str_ori = ' '.join([x.split(" ")[0] for x in line])
                    # str_new = ' '.join([x.split(" ")[0] for x in new_line])
                    # log.info("ORIGINAL: " + str_ori)
                    # log.info("NEW_LINE: " + str_new)

                sentence = self._convert_lines_to_sentence(new_line)

            # set sentence context using partials
            sentence._position_in_dataset = (self, index)

        return sentence


class CONLL_03_Shuf(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = False,
            **corpusargs,
    ):
        """
        shuffled conll03 dataset with its inner label dictionary
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

        super(CONLL_03_Shuf, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            **corpusargs,
        )

# from torch.utils.data.dataloader import DataLoader
if __name__ == "__main__":
    data = CONLL_03_Shuf("/research/d4/gds/yczhang21/project/CSCI5640_NLP")
    print(data.train)
    dataloader = DataLoader(data.train)
    iter = 0
    for batch in dataloader:
        # print(batch[0]._embeddings)

        # for token in batch[0]:
        #     print(token, token.whitespace_after, token.get_tag("ner"))
        iter += 1
        if iter == 10:
            break
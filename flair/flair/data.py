import logging
import re
from abc import abstractmethod, ABC
from collections import Counter
from collections import defaultdict
from operator import itemgetter
from typing import List, Dict, Union, Optional
import numpy as np
from deprecated import deprecated
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset, Subset

import flair
from flair.file_utils import Tqdm

log = logging.getLogger("flair")


class Dictionary:
    """
    This class holds a dictionary that maps strings to IDs, used to generate one-hot encodings of strings.
    """

    def __init__(self, add_unk=True):
        # init dictionaries
        self.item2idx: Dict[str, int] = {}
        self.idx2item: List[str] = []
        self.add_unk = add_unk
        # in order to deal with unknown tokens, add <unk>
        if add_unk:
            self.add_item("<unk>")

    def remove_item(self, item: str):

        item = item.encode("utf-8")
        if item in self.item2idx:
            self.idx2item.remove(item)
            del self.item2idx[item]

    def add_item(self, item: str) -> int:
        """
        add string - if already in dictionary returns its ID. if not in dictionary, it will get a new ID.
        :param item: a string for which to assign an id.
        :return: ID of string
        """
        item = item.encode("utf-8")
        if item not in self.item2idx:
            self.idx2item.append(item)
            self.item2idx[item] = len(self.idx2item) - 1
        return self.item2idx[item]

    def get_idx_for_item(self, item: str) -> int:
        """
        returns the ID of the string, otherwise 0
        :param item: string for which ID is requested
        :return: ID of string, otherwise 0
        """
        item_encoded = item.encode("utf-8")
        if item_encoded in self.item2idx.keys():
            return self.item2idx[item_encoded]
        elif self.add_unk:
            return 0
        else:
            log.error(f"The string '{item}' is not in dictionary! Dictionary contains only: {self.get_items()}")
            log.error(
                "You can create a Dictionary that handles unknown items with an <unk>-key by setting add_unk = True in the construction.")
            raise IndexError

    def get_idx_for_items(self, items: List[str]) -> List[int]:
        """
        returns the IDs for each item of the list of string, otherwise 0 if not found
        :param items: List of string for which IDs are requested
        :return: List of ID of strings
        """
        if not hasattr(self, "item2idx_not_encoded"):
            d = dict(
                [(key.decode("UTF-8"), value) for key, value in self.item2idx.items()]
            )
            self.item2idx_not_encoded = defaultdict(int, d)

        if not items:
            return []
        results = itemgetter(*items)(self.item2idx_not_encoded)
        if isinstance(results, int):
            return [results]
        return list(results)

    def get_items(self) -> List[str]:
        items = []
        for item in self.idx2item:
            items.append(item.decode("UTF-8"))
        return items

    def __len__(self) -> int:
        return len(self.idx2item)

    def get_item_for_index(self, idx):
        return self.idx2item[idx].decode("UTF-8")

    def save(self, savefile):
        import pickle

        with open(savefile, "wb") as f:
            mappings = {"idx2item": self.idx2item, "item2idx": self.item2idx}
            pickle.dump(mappings, f)

    def __setstate__(self, d):
        self.__dict__ = d
        # set 'add_unk' if the dictionary was created with a version of Flair older than 0.9
        if 'add_unk' not in self.__dict__.keys():
            self.__dict__['add_unk'] = True if b'<unk>' in self.__dict__['idx2item'] else False

    @classmethod
    def load_from_file(cls, filename: str):
        import pickle

        f = open(filename, "rb")
        mappings = pickle.load(f, encoding="latin1")
        idx2item = mappings["idx2item"]
        item2idx = mappings["item2idx"]
        f.close()

        # set 'add_unk' depending on whether <unk> is a key
        add_unk = True if b'<unk>' in idx2item else False

        dictionary: Dictionary = Dictionary(add_unk=add_unk)
        dictionary.item2idx = item2idx
        dictionary.idx2item = idx2item
        return dictionary

    @classmethod
    def load(cls, name: str):
        from flair.file_utils import cached_path
        hu_path: str = "https://flair.informatik.hu-berlin.de/resources/characters"
        if name == "chars" or name == "common-chars":
            char_dict = cached_path(f"{hu_path}/common_characters", cache_dir="datasets")
            return Dictionary.load_from_file(char_dict)

        if name == "chars-large" or name == "common-chars-large":
            char_dict = cached_path(f"{hu_path}/common_characters_large", cache_dir="datasets")
            return Dictionary.load_from_file(char_dict)

        if name == "chars-xl" or name == "common-chars-xl":
            char_dict = cached_path(f"{hu_path}/common_characters_xl", cache_dir="datasets")
            return Dictionary.load_from_file(char_dict)

        return Dictionary.load_from_file(name)

    def __str__(self):
        tags = ', '.join(self.get_item_for_index(i) for i in range(min(len(self), 50)))
        return f"Dictionary with {len(self)} tags: {tags}"


class Label:
    """
    This class represents a label. Each label has a value and optionally a confidence score. The
    score needs to be between 0.0 and 1.0. Default value for the score is 1.0.
    """

    def __init__(self, value: str, score: float = 1.0):
        self._value = value
        self._score = score
        super().__init__()

    def set_value(self, value: str, score: float = 1.0):
        self.value = value
        self.score = score

    def spawn(self, value: str, score: float = 1.0):
        return Label(value, score)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not value and value != "":
            raise ValueError(
                "Incorrect label value provided. Label value needs to be set."
            )
        else:
            self._value = value

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        self._score = score

    def to_dict(self):
        return {"value": self.value, "confidence": self.score}

    def __str__(self):
        return f"{self._value} ({round(self._score, 4)})"

    def __repr__(self):
        return f"{self._value} ({round(self._score, 4)})"

    def __eq__(self, other):
        return self.value == other.value and self.score == other.score

    @property
    def identifier(self):
        return ""


class SpanLabel(Label):
    def __init__(self, span, value: str, score: float = 1.0):
        super().__init__(value, score)
        self.span = span

    def spawn(self, value: str, score: float = 1.0):
        return SpanLabel(self.span, value, score)

    def __str__(self):
        return f"{self._value} [{self.span.id_text}] ({round(self._score, 4)})"

    def __repr__(self):
        return f"{self._value} [{self.span.id_text}] ({round(self._score, 4)})"

    def __len__(self):
        return len(self.span)

    def __eq__(self, other):
        return self.value == other.value and self.score == other.score and self.span.id_text == other.span.id_text

    @property
    def identifier(self):
        return f"{self.span.id_text}"


class RelationLabel(Label):
    def __init__(self, head, tail, value: str, score: float = 1.0):
        super().__init__(value, score)
        self.head = head
        self.tail = tail

    def spawn(self, value: str, score: float = 1.0):
        return RelationLabel(self.head, self.tail, value, score)

    def __str__(self):
        return f"{self._value} [{self.head.id_text} -> {self.tail.id_text}] ({round(self._score, 4)})"

    def __repr__(self):
        return f"{self._value} from {self.head.id_text} -> {self.tail.id_text} ({round(self._score, 4)})"

    def __len__(self):
        return len(self.head) + len(self.tail)

    def __eq__(self, other):
        return self.value == other.value \
               and self.score == other.score \
               and self.head.id_text == other.head.id_text \
               and self.tail.id_text == other.tail.id_text

    @property
    def identifier(self):
        return f"{self.head.id_text} -> {self.tail.id_text}"


class DataPoint:
    """
    This is the parent class of all data points in Flair (including Token, Sentence, Image, etc.). Each DataPoint
    must be embeddable (hence the abstract property embedding() and methods to() and clear_embeddings()). Also,
    each DataPoint may have Labels in several layers of annotation (hence the functions add_label(), get_labels()
    and the property 'label')
    """

    def __init__(self):
        self.annotation_layers = {}

    @property
    @abstractmethod
    def embedding(self):
        pass

    @abstractmethod
    def to(self, device: str, pin_memory: bool = False):
        pass

    @abstractmethod
    def clear_embeddings(self, embedding_names: List[str] = None):
        pass

    def add_label(self, typename: str, value: str, score: float = 1.):

        if typename not in self.annotation_layers:
            self.annotation_layers[typename] = [Label(value, score)]
        else:
            self.annotation_layers[typename].append(Label(value, score))

        return self

    def add_complex_label(self, typename: str, label: Label):

        if typename in self.annotation_layers and label in self.annotation_layers[typename]:
            return self

        if typename not in self.annotation_layers:
            self.annotation_layers[typename] = [label]
        else:
            self.annotation_layers[typename].append(label)

        return self

    def set_label(self, typename: str, value: str, score: float = 1.):
        self.annotation_layers[typename] = [Label(value, score)]
        return self

    def remove_labels(self, typename: str):
        if typename in self.annotation_layers.keys():
            del self.annotation_layers[typename]

    def get_labels(self, typename: str = None):
        if typename is None:
            return self.labels

        return self.annotation_layers[typename] if typename in self.annotation_layers else []

    @property
    def labels(self) -> List[Label]:
        all_labels = []
        for key in self.annotation_layers.keys():
            all_labels.extend(self.annotation_layers[key])
        return all_labels


class DataPair(DataPoint):
    def __init__(self, first: DataPoint, second: DataPoint):
        super().__init__()
        self.first = first
        self.second = second

    def to(self, device: str, pin_memory: bool = False):
        self.first.to(device, pin_memory)
        self.second.to(device, pin_memory)

    def clear_embeddings(self, embedding_names: List[str] = None):
        self.first.clear_embeddings(embedding_names)
        self.second.clear_embeddings(embedding_names)

    @property
    def embedding(self):
        return torch.cat([self.first.embedding, self.second.embedding])

    def __str__(self):
        return f"DataPair:\n − First {self.first}\n − Second {self.second}\n − Labels: {self.labels}"

    def to_plain_string(self):
        return f"DataPair: First {self.first}  ||  Second {self.second}"

    def to_original_text(self):
        return f"{self.first.to_original_text()} || {self.second.to_original_text()}"

    def __len__(self):
        return len(self.first) + len(self.second)


class Token(DataPoint):
    """
    This class represents one word in a tokenized sentence. Each token may have any number of tags. It may also point
    to its head in a dependency tree.
    """

    def __init__(
            self,
            text: str,
            idx: int = None,
            head_id: int = None,
            whitespace_after: bool = True,
            start_position: int = None,
    ):
        super().__init__()

        self.text: str = text
        self.idx: int = idx
        self.head_id: int = head_id
        self.whitespace_after: bool = whitespace_after

        self.start_pos = start_position
        self.end_pos = (
            start_position + len(text) if start_position is not None else None
        )

        self.sentence: Sentence = None
        self._embeddings: Dict = {}
        self.tags_proba_dist: Dict[str, List[Label]] = {}

    def add_tag_label(self, tag_type: str, tag: Label):
        self.set_label(tag_type, tag.value, tag.score)

    def add_tags_proba_dist(self, tag_type: str, tags: List[Label]):
        self.tags_proba_dist[tag_type] = tags

    def add_tag(self, tag_type: str, tag_value: str, confidence=1.0):
        self.set_label(tag_type, tag_value, confidence)

    def get_tag(self, label_type):
        if len(self.get_labels(label_type)) == 0: return Label('')
        return self.get_labels(label_type)[0]

    def get_tags_proba_dist(self, tag_type: str) -> List[Label]:
        if tag_type in self.tags_proba_dist:
            return self.tags_proba_dist[tag_type]
        return []

    def get_head(self):
        return self.sentence.get_token(self.head_id)

    def set_embedding(self, name: str, vector: torch.tensor):
        device = flair.device
        if (flair.embedding_storage_mode == "cpu") and len(self._embeddings.keys()) > 0:
            device = next(iter(self._embeddings.values())).device
        if device != vector.device:
            vector = vector.to(device)
        self._embeddings[name] = vector

    def to(self, device: str, pin_memory: bool = False):
        for name, vector in self._embeddings.items():
            if str(vector.device) != str(device):
                if pin_memory:
                    self._embeddings[name] = vector.to(
                        device, non_blocking=True
                    ).pin_memory()
                else:
                    self._embeddings[name] = vector.to(device, non_blocking=True)

    def clear_embeddings(self, embedding_names: List[str] = None):
        if embedding_names is None:
            self._embeddings: Dict = {}
        else:
            for name in embedding_names:
                if name in self._embeddings.keys():
                    del self._embeddings[name]

    def get_each_embedding(self, embedding_names: Optional[List[str]] = None) -> torch.tensor:
        embeddings = []
        for embed in sorted(self._embeddings.keys()):
            if embedding_names and embed not in embedding_names: continue
            embed = self._embeddings[embed].to(flair.device)
            if (flair.embedding_storage_mode == "cpu") and embed.device != flair.device:
                embed = embed.to(flair.device)
            embeddings.append(embed)
        return embeddings

    def get_embedding(self, names: Optional[List[str]] = None) -> torch.tensor:
        embeddings = self.get_each_embedding(names)

        if embeddings:
            return torch.cat(embeddings, dim=0)

        return torch.tensor([], device=flair.device)

    @property
    def start_position(self) -> int:
        return self.start_pos

    @property
    def end_position(self) -> int:
        return self.end_pos

    @property
    def embedding(self):
        return self.get_embedding()

    def __str__(self) -> str:
        return (
            "Token: {} {}".format(self.idx, self.text)
            if self.idx is not None
            else "Token: {}".format(self.text)
        )

    def __repr__(self) -> str:
        return (
            "Token: {} {}".format(self.idx, self.text)
            if self.idx is not None
            else "Token: {}".format(self.text)
        )


class Span(DataPoint):
    """
    This class represents one textual span consisting of Tokens.
    """

    def __init__(self, tokens: List[Token]):

        super().__init__()

        self.tokens = tokens
        self.start_pos = None
        self.end_pos = None

        if tokens:
            self.start_pos = tokens[0].start_position
            self.end_pos = tokens[len(tokens) - 1].end_position

    @property
    def text(self) -> str:
        return " ".join([t.text for t in self.tokens])

    def to_original_text(self) -> str:
        pos = self.tokens[0].start_pos
        if pos is None:
            return " ".join([t.text for t in self.tokens])
        str = ""
        for t in self.tokens:
            while t.start_pos > pos:
                str += " "
                pos += 1

            str += t.text
            pos += len(t.text)

        return str

    def to_plain_string(self):
        plain = ""
        for token in self.tokens:
            plain += token.text
            if token.whitespace_after:
                plain += " "
        return plain.rstrip()

    def to_dict(self):
        return {
            "text": self.to_original_text(),
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "labels": self.labels,
        }

    def __str__(self) -> str:
        ids = ",".join([str(t.idx) for t in self.tokens])
        label_string = " ".join([str(label) for label in self.labels])
        labels = f'   [− Labels: {label_string}]' if self.labels else ""
        return (
            'Span [{}]: "{}"{}'.format(ids, self.text, labels)
        )

    @property
    def id_text(self) -> str:
        return f"{' '.join([t.text for t in self.tokens])} ({','.join([str(t.idx) for t in self.tokens])})"

    def __repr__(self) -> str:
        ids = ",".join([str(t.idx) for t in self.tokens])
        return (
            '<{}-span ({}): "{}">'.format(self.tag, ids, self.text)
            if len(self.labels) > 0
            else '<span ({}): "{}">'.format(ids, self.text)
        )

    def __getitem__(self, idx: int) -> Token:
        return self.tokens[idx]

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def tag(self):
        return self.labels[0].value

    @property
    def score(self):
        return self.labels[0].score

    @property
    def position_string(self):
        return '-'.join([str(token.idx) for token in self])


class Tokenizer(ABC):
    r"""An abstract class representing a :class:`Tokenizer`.

    Tokenizers are used to represent algorithms and models to split plain text into
    individual tokens / words. All subclasses should overwrite :meth:`tokenize`, which
    splits the given plain text into tokens. Moreover, subclasses may overwrite
    :meth:`name`, returning a unique identifier representing the tokenizer's
    configuration.
    """

    @abstractmethod
    def tokenize(self, text: str) -> List[Token]:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self.__class__.__name__


class Sentence(DataPoint):
    """
       A Sentence is a list of tokens and is used to represent a sentence or text fragment.
    """

    def __init__(
            self,
            text: Union[str, List[str]] = None,
            use_tokenizer: Union[bool, Tokenizer] = True,
            language_code: str = None,
            start_position: int = None
    ):
        """
        Class to hold all meta related to a text (tokens, predictions, language code, ...)
        :param text: original string (sentence), or a list of string tokens (words)
        :param use_tokenizer: a custom tokenizer (default is :class:`SpaceTokenizer`)
            more advanced options are :class:`SegTokTokenizer` to use segtok or :class:`SpacyTokenizer`
            to use Spacy library if available). Check the implementations of abstract class Tokenizer or
            implement your own subclass (if you need it). If instead of providing a Tokenizer, this parameter
            is just set to True (deprecated), :class:`SegtokTokenizer` will be used.
        :param language_code: Language of the sentence
        :param start_position: Start char offset of the sentence in the superordinate document
        """
        super().__init__()

        self.tokens: List[Token] = []

        self._embeddings: Dict = {}
        # MODIFIED：build entity relation
        self.entity_relation = []
        self.language_code: str = language_code

        self.start_pos = start_position
        self.end_pos = (
            start_position + len(text) if start_position is not None else None
        )

        if isinstance(use_tokenizer, Tokenizer):
            tokenizer = use_tokenizer
        elif hasattr(use_tokenizer, "__call__"):
            from flair.tokenization import TokenizerWrapper
            tokenizer = TokenizerWrapper(use_tokenizer)
        elif type(use_tokenizer) == bool:
            from flair.tokenization import SegtokTokenizer, SpaceTokenizer
            tokenizer = SegtokTokenizer() if use_tokenizer else SpaceTokenizer()
        else:
            raise AssertionError("Unexpected type of parameter 'use_tokenizer'. " +
                                 "Parameter should be bool, Callable[[str], List[Token]] (deprecated), Tokenizer")

        # if text is passed, instantiate sentence with tokens (words)
        if text is not None:
            if isinstance(text, (list, tuple)):
                [self.add_token(self._restore_windows_1252_characters(token))
                 for token in text]
            else:
                text = self._restore_windows_1252_characters(text)
                [self.add_token(token) for token in tokenizer.tokenize(text)]

        # log a warning if the dataset is empty
        if text == "":
            log.warning(
                "Warning: An empty Sentence was created! Are there empty strings in your dataset?"
            )

        self.tokenized = None

        # some sentences represent a document boundary (but most do not)
        self.is_document_boundary: bool = False

    def get_token(self, token_id: int) -> Token:
        for token in self.tokens:
            if token.idx == token_id:
                return token

    def add_token(self, token: Union[Token, str]):

        if type(token) is str:
            token = Token(token)

        token.text = token.text.replace('\u200c', '')
        token.text = token.text.replace('\u200b', '')
        token.text = token.text.replace('\ufe0f', '')
        token.text = token.text.replace('\ufeff', '')

        # data with zero-width characters cannot be handled
        if token.text == '':
            return

        self.tokens.append(token)

        # set token idx if not set
        token.sentence = self
        if token.idx is None:
            token.idx = len(self.tokens)

    def get_label_names(self):
        label_names = []
        for label in self.labels:
            label_names.append(label.value)
        return label_names

    def _add_spans_internal(self, spans: List[Span], label_type: str, min_score):

        current_span = []

        tags = defaultdict(lambda: 0.0)

        previous_tag_value: str = "O"
        for token in self:

            tag: Label = token.get_tag(label_type)
            tag_value = tag.value

            # non-set tags are OUT tags
            if tag_value == "" or tag_value == "O" or tag_value == "_":
                tag_value = "O-"

            # anything that is not a BIOES tag is a SINGLE tag
            if tag_value[0:2] not in ["B-", "I-", "O-", "E-", "S-"]:
                tag_value = "S-" + tag_value

            # anything that is not OUT is IN
            in_span = False
            if tag_value[0:2] not in ["O-"]:
                in_span = True

            # single and begin tags start a new span
            starts_new_span = False
            if tag_value[0:2] in ["B-", "S-"]:
                starts_new_span = True

            if (
                    previous_tag_value[0:2] in ["S-"]
                    and previous_tag_value[2:] != tag_value[2:]
                    and in_span
            ):
                starts_new_span = True

            if (starts_new_span or not in_span) and len(current_span) > 0:
                scores = [t.get_labels(label_type)[0].score for t in current_span]
                span_score = sum(scores) / len(scores)
                if span_score > min_score:
                    span = Span(current_span)
                    span.add_label(
                        typename=label_type,
                        value=sorted(tags.items(), key=lambda k_v: k_v[1], reverse=True)[0][0],
                        score=span_score)
                    spans.append(span)

                current_span = []
                tags = defaultdict(lambda: 0.0)

            if in_span:
                current_span.append(token)
                weight = 1.1 if starts_new_span else 1.0
                tags[tag_value[2:]] += weight

            # remember previous tag
            previous_tag_value = tag_value

        if len(current_span) > 0:
            scores = [t.get_labels(label_type)[0].score for t in current_span]
            span_score = sum(scores) / len(scores)
            if span_score > min_score:
                span = Span(current_span)
                span.add_label(
                    typename=label_type,
                    value=sorted(tags.items(), key=lambda k_v: k_v[1], reverse=True)[0][0],
                    score=span_score)
                spans.append(span)

        return spans

    def get_spans(self, label_type: Optional[str] = None, min_score=-1) -> List[Span]:

        spans: List[Span] = []

        # if label type is explicitly specified, get spans for this label type
        if label_type:
            return self._add_spans_internal(spans, label_type, min_score)

        # else determine all label types in sentence and get all spans
        label_types = []
        for token in self:
            for annotation in token.annotation_layers.keys():
                if annotation not in label_types: label_types.append(annotation)

        for label_type in label_types:
            self._add_spans_internal(spans, label_type, min_score)
        return spans

    @property
    def embedding(self):
        return self.get_embedding()

    def set_embedding(self, name: str, vector: torch.tensor):
        device = flair.device
        if (flair.embedding_storage_mode == "cpu") and len(self._embeddings.keys()) > 0:
            device = next(iter(self._embeddings.values())).device
        if device != vector.device:
            vector = vector.to(device)
        self._embeddings[name] = vector

    def get_embedding(self, names: Optional[List[str]] = None) -> torch.tensor:
        embeddings = []
        for embed in sorted(self._embeddings.keys()):
            if names and embed not in names: continue
            embedding = self._embeddings[embed]
            embeddings.append(embedding)

        if embeddings:
            return torch.cat(embeddings, dim=0)

        return torch.Tensor()

    def to(self, device: str, pin_memory: bool = False):

        # move sentence embeddings to device
        for name, vector in self._embeddings.items():
            if str(vector.device) != str(device):
                if pin_memory:
                    self._embeddings[name] = vector.to(
                        device, non_blocking=True
                    ).pin_memory()
                else:
                    self._embeddings[name] = vector.to(device, non_blocking=True)

        # move token embeddings to device
        for token in self:
            token.to(device, pin_memory)

    def clear_embeddings(self, embedding_names: List[str] = None):

        # clear sentence embeddings
        if embedding_names is None:
            self._embeddings: Dict = {}
        else:
            for name in embedding_names:
                if name in self._embeddings.keys():
                    del self._embeddings[name]

        # clear token embeddings
        for token in self:
            token.clear_embeddings(embedding_names)

    def to_tagged_string(self, main_tag=None) -> str:
        list = []
        for token in self.tokens:
            list.append(token.text)

            tags: List[str] = []
            for label_type in token.annotation_layers.keys():

                if main_tag is not None and main_tag != label_type:
                    continue

                if token.get_labels(label_type)[0].value == "O":
                    continue
                if token.get_labels(label_type)[0].value == "_":
                    continue

                tags.append(token.get_labels(label_type)[0].value)
            all_tags = "<" + "/".join(tags) + ">"
            if all_tags != "<>":
                list.append(all_tags)
        return " ".join(list)

    def to_tokenized_string(self) -> str:

        if self.tokenized is None:
            self.tokenized = " ".join([t.text for t in self.tokens])

        return self.tokenized

    def to_plain_string(self):
        plain = ""
        for token in self.tokens:
            plain += token.text
            if token.whitespace_after:
                plain += " "
        return plain.rstrip()
        
    # MODIFIED：build entity relation
    def cal_entity_relation(self, span=3, label_type='ner'):
        self.entity_relation = np.zeros((len(self), span*2 + 1))
        for i, token in enumerate(self.tokens):
            tag_val = token.get_tag(label_type).value
            if len(tag_val) < 5:
                # for j in range(span*2 + 1):
                #     self.entity_relation[i][j] = -1
                continue
            else:
                self.entity_relation[i][3] = 1
                if tag_val[0] == "S":
                    continue

            # forward search
            for j in range(1, 4):
                if i-j < 0:
                    break
                tag_val_ref = self.tokens[i-j].get_tag(label_type).value
                # same semantic meaning:
                if tag_val_ref[1:] == tag_val[1:]:
                    if tag_val_ref[0] == "B":
                        self.entity_relation[i][3-j] = 1
                        break
                    elif tag_val_ref[0] == "I":
                        self.entity_relation[i][3-j] = 1
                    else:
                        break
                else:
                    break

            # backward search
            for j in range(1, 4):
                if i+j >= len(self):
                    break
                tag_val_ref = self.tokens[i+j].get_tag(label_type).value
                # same semantic meaning:
                if tag_val_ref[1:] == tag_val[1:]:
                    if tag_val_ref[0] == "E":
                        self.entity_relation[i][3+j] = 1
                        break
                    elif tag_val_ref[0] == "I":
                        self.entity_relation[i][3+j] = 1
                    else:
                        break
                else:
                    break

        # print(self.entity_relation, self)
                

    def convert_tag_scheme(self, tag_type: str = "ner", target_scheme: str = "iob"):

        tags: List[Label] = []
        for token in self.tokens:
            tags.append(token.get_tag(tag_type))

        if target_scheme == "iob":
            iob2(tags)

        if target_scheme == "iobes":
            iob2(tags)
            tags = iob_iobes(tags)

        # MODIFIED：target scheme.
        if target_scheme == "i-only":
            tags = iob_i_only(tags)

        for index, tag in enumerate(tags):
            self.tokens[index].set_label(tag_type, tag)
    
    def infer_space_after(self):
        """
        Heuristics in case you wish to infer whitespace_after values for tokenized text. This is useful for some old NLP
        tasks (such as CoNLL-03 and CoNLL-2000) that provide only tokenized data with no info of original whitespacing.
        :return:
        """
        last_token = None
        quote_count: int = 0
        # infer whitespace after field

        for token in self.tokens:
            if token.text == '"':
                quote_count += 1
                if quote_count % 2 != 0:
                    token.whitespace_after = False
                elif last_token is not None:
                    last_token.whitespace_after = False

            if last_token is not None:

                if token.text in [".", ":", ",", ";", ")", "n't", "!", "?"]:
                    last_token.whitespace_after = False

                if token.text.startswith("'"):
                    last_token.whitespace_after = False

            if token.text in ["("]:
                token.whitespace_after = False

            last_token = token
        return self

    def to_original_text(self) -> str:
        if len(self.tokens) > 0 and (self.tokens[0].start_pos is None):
            return " ".join([t.text for t in self.tokens])
        str = ""
        pos = 0
        for t in self.tokens:
            while t.start_pos > pos:
                str += " "
                pos += 1

            str += t.text
            pos += len(t.text)

        return str

    def to_dict(self, tag_type: str = None):
        labels = []
        entities = []

        if tag_type:
            entities = [span.to_dict() for span in self.get_spans(tag_type)]
        if self.labels:
            labels = [l.to_dict() for l in self.labels]

        return {"text": self.to_original_text(), "labels": labels, "entities": entities}

    def __getitem__(self, idx: int) -> Token:
        return self.tokens[idx]

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def __repr__(self):
        tagged_string = self.to_tagged_string()
        tokenized_string = self.to_tokenized_string()

        # add Sentence labels to output if they exist
        sentence_labels = f"  − Sentence-Labels: {self.annotation_layers}" if self.annotation_layers != {} else ""

        # add Token labels to output if they exist
        token_labels = f'  − Token-Labels: "{tagged_string}"' if tokenized_string != tagged_string else ""

        return f'Sentence: "{tokenized_string}"   [− Tokens: {len(self)}{token_labels}{sentence_labels}]'

    def __copy__(self):
        s = Sentence()
        for token in self.tokens:
            nt = Token(token.text)
            for tag_type in token.tags:
                nt.add_label(
                    tag_type,
                    token.get_tag(tag_type).value,
                    token.get_tag(tag_type).score,
                )

            s.add_token(nt)
        return s

    def __str__(self) -> str:

        tagged_string = self.to_tagged_string()
        tokenized_string = self.to_tokenized_string()

        # add Sentence labels to output if they exist
        sentence_labels = f"  − Sentence-Labels: {self.annotation_layers}" if self.annotation_layers != {} else ""

        # add Token labels to output if they exist
        token_labels = f'  − Token-Labels: "{tagged_string}"' if tokenized_string != tagged_string else ""

        return f'Sentence: "{tokenized_string}"   [− Tokens: {len(self)}{token_labels}{sentence_labels}]'

    def get_language_code(self) -> str:
        if self.language_code is None:
            import langdetect

            try:
                self.language_code = langdetect.detect(self.to_plain_string())
            except:
                self.language_code = "en"

        return self.language_code

    @staticmethod
    def _restore_windows_1252_characters(text: str) -> str:
        def to_windows_1252(match):
            try:
                return bytes([ord(match.group(0))]).decode("windows-1252")
            except UnicodeDecodeError:
                # No character at the corresponding code point: remove it
                return ""

        return re.sub(r"[\u0080-\u0099]", to_windows_1252, text)

    def next_sentence(self):
        """
        Get the next sentence in the document (works only if context is set through dataloader or elsewhere)
        :return: next Sentence in document if set, otherwise None
        """
        if '_next_sentence' in self.__dict__.keys():
            return self._next_sentence

        if '_position_in_dataset' in self.__dict__.keys():
            dataset = self._position_in_dataset[0]
            index = self._position_in_dataset[1] + 1
            if index < len(dataset):
                return dataset[index]

        return None

    def previous_sentence(self):
        """
        Get the previous sentence in the document (works only if context is set through dataloader or elsewhere)
        :return: previous Sentence in document if set, otherwise None
        """
        if '_previous_sentence' in self.__dict__.keys():
            return self._previous_sentence

        if '_position_in_dataset' in self.__dict__.keys():
            dataset = self._position_in_dataset[0]
            index = self._position_in_dataset[1] - 1
            if index >= 0:
                return dataset[index]

        return None

    def is_context_set(self) -> bool:
        """
        Return True or False depending on whether context is set (for instance in dataloader or elsewhere)
        :return: True if context is set, else False
        """
        return '_previous_sentence' in self.__dict__.keys() or '_position_in_dataset' in self.__dict__.keys()

    def get_labels(self, label_type: str = None):

        # TODO: crude hack - replace with something better
        if label_type:
            spans = self.get_spans(label_type)
            for span in spans:
                self.add_complex_label(label_type, label=SpanLabel(span, span.tag, span.score))

        if label_type is None:
            return self.labels

        return self.annotation_layers[label_type] if label_type in self.annotation_layers else []


class Image(DataPoint):

    def __init__(self, data=None, imageURL=None):
        super().__init__()

        self.data = data
        self._embeddings: Dict = {}
        self.imageURL = imageURL

    @property
    def embedding(self):
        return self.get_embedding()

    def __str__(self):

        image_repr = self.data.size() if self.data else ""
        image_url = self.imageURL if self.imageURL else ""

        return f"Image: {image_repr} {image_url}"

    def get_embedding(self) -> torch.tensor:
        embeddings = [
            self._embeddings[embed] for embed in sorted(self._embeddings.keys())
        ]

        if embeddings:
            return torch.cat(embeddings, dim=0)

        return torch.tensor([], device=flair.device)

    def set_embedding(self, name: str, vector: torch.tensor):
        device = flair.device
        if (flair.embedding_storage_mode == "cpu") and len(self._embeddings.keys()) > 0:
            device = next(iter(self._embeddings.values())).device
        if device != vector.device:
            vector = vector.to(device)
        self._embeddings[name] = vector

    def to(self, device: str, pin_memory: bool = False):
        for name, vector in self._embeddings.items():
            if str(vector.device) != str(device):
                if pin_memory:
                    self._embeddings[name] = vector.to(
                        device, non_blocking=True
                    ).pin_memory()
                else:
                    self._embeddings[name] = vector.to(device, non_blocking=True)

    def clear_embeddings(self, embedding_names: List[str] = None):
        if embedding_names is None:
            self._embeddings: Dict = {}
        else:
            for name in embedding_names:
                if name in self._embeddings.keys():
                    del self._embeddings[name]


class FlairDataset(Dataset):
    @abstractmethod
    def is_in_memory(self) -> bool:
        pass


class Corpus:
    def __init__(
            self,
            train: FlairDataset = None,
            dev: FlairDataset = None,
            test: FlairDataset = None,
            name: str = "corpus",
            sample_missing_splits: Union[bool, str] = True,
    ):
        # set name
        self.name: str = name
        
        # abort if no data is provided
        if not train and not dev and not test:
            raise RuntimeError('No data provided when initializing corpus object.')

        # sample test data from train if none is provided
        if test is None and sample_missing_splits and train and not sample_missing_splits == 'only_dev':
            train_length = len(train)
            test_size: int = round(train_length / 10)
            splits = randomly_split_into_two_datasets(train, test_size)
            test = splits[0]
            train = splits[1]

        # sample dev data from train if none is provided
        if dev is None and sample_missing_splits and train and not sample_missing_splits == 'only_test':
            train_length = len(train)
            dev_size: int = round(train_length / 10)
            splits = randomly_split_into_two_datasets(train, dev_size)
            dev = splits[0]
            train = splits[1]

        # set train dev and test data
        self._train: FlairDataset = train
        self._test: FlairDataset = test
        self._dev: FlairDataset = dev

    @property
    def train(self) -> FlairDataset:
        return self._train

    @property
    def dev(self) -> FlairDataset:
        return self._dev

    @property
    def test(self) -> FlairDataset:
        return self._test

    def downsample(self, percentage: float = 0.1, downsample_train=True, downsample_dev=True, downsample_test=True):

        if downsample_train and self._train:
            self._train = self._downsample_to_proportion(self.train, percentage)

        if downsample_dev and self._dev:
            self._dev = self._downsample_to_proportion(self.dev, percentage)

        if downsample_test and self._test:
            self._test = self._downsample_to_proportion(self.test, percentage)

        return self

    def filter_empty_sentences(self):
        log.info("Filtering empty sentences")
        self._train = Corpus._filter_empty_sentences(self._train)
        self._test = Corpus._filter_empty_sentences(self._test)
        self._dev = Corpus._filter_empty_sentences(self._dev)
        log.info(self)

    def filter_long_sentences(self, max_charlength: int):
        log.info("Filtering long sentences")
        self._train = Corpus._filter_long_sentences(self._train, max_charlength)
        self._test = Corpus._filter_long_sentences(self._test, max_charlength)
        self._dev = Corpus._filter_long_sentences(self._dev, max_charlength)
        log.info(self)

    @staticmethod
    def _filter_long_sentences(dataset, max_charlength: int) -> Dataset:

        # find out empty sentence indices
        empty_sentence_indices = []
        non_empty_sentence_indices = []
        index = 0

        from flair.datasets import DataLoader

        for batch in DataLoader(dataset):
            for sentence in batch:
                if len(sentence.to_plain_string()) > max_charlength:
                    empty_sentence_indices.append(index)
                else:
                    non_empty_sentence_indices.append(index)
                index += 1

        # create subset of non-empty sentence indices
        subset = Subset(dataset, non_empty_sentence_indices)

        return subset

    @staticmethod
    def _filter_empty_sentences(dataset) -> Dataset:

        # find out empty sentence indices
        empty_sentence_indices = []
        non_empty_sentence_indices = []
        index = 0

        from flair.datasets import DataLoader

        for batch in DataLoader(dataset):
            for sentence in batch:
                if len(sentence) == 0:
                    empty_sentence_indices.append(index)
                else:
                    non_empty_sentence_indices.append(index)
                index += 1

        # create subset of non-empty sentence indices
        subset = Subset(dataset, non_empty_sentence_indices)

        return subset

    def make_vocab_dictionary(self, max_tokens=-1, min_freq=1) -> Dictionary:
        """
        Creates a dictionary of all tokens contained in the corpus.
        By defining `max_tokens` you can set the maximum number of tokens that should be contained in the dictionary.
        If there are more than `max_tokens` tokens in the corpus, the most frequent tokens are added first.
        If `min_freq` is set the a value greater than 1 only tokens occurring more than `min_freq` times are considered
        to be added to the dictionary.
        :param max_tokens: the maximum number of tokens that should be added to the dictionary (-1 = take all tokens)
        :param min_freq: a token needs to occur at least `min_freq` times to be added to the dictionary (-1 = there is no limitation)
        :return: dictionary of tokens
        """
        tokens = self._get_most_common_tokens(max_tokens, min_freq)

        vocab_dictionary: Dictionary = Dictionary()
        for token in tokens:
            vocab_dictionary.add_item(token)

        return vocab_dictionary

    def _get_most_common_tokens(self, max_tokens, min_freq) -> List[str]:
        tokens_and_frequencies = Counter(self._get_all_tokens())
        tokens_and_frequencies = tokens_and_frequencies.most_common()

        tokens = []
        for token, freq in tokens_and_frequencies:
            if (min_freq != -1 and freq < min_freq) or (
                    max_tokens != -1 and len(tokens) == max_tokens
            ):
                break
            tokens.append(token)
        return tokens

    def _get_all_tokens(self) -> List[str]:
        tokens = list(map((lambda s: s.tokens), self.train))
        tokens = [token for sublist in tokens for token in sublist]
        return list(map((lambda t: t.text), tokens))

    @staticmethod
    def _downsample_to_proportion(dataset: Dataset, proportion: float):

        sampled_size: int = round(len(dataset) * proportion)
        splits = randomly_split_into_two_datasets(dataset, sampled_size)
        return splits[0]

    def obtain_statistics(
            self, label_type: str = None, pretty_print: bool = True
    ) -> dict:
        """
        Print statistics about the class distribution (only labels of sentences are taken into account) and sentence
        sizes.
        """
        json_string = {
            "TRAIN": self._obtain_statistics_for(self.train, "TRAIN", label_type),
            "TEST": self._obtain_statistics_for(self.test, "TEST", label_type),
            "DEV": self._obtain_statistics_for(self.dev, "DEV", label_type),
        }
        if pretty_print:
            import json

            json_string = json.dumps(json_string, indent=4)
        return json_string

    @staticmethod
    def _obtain_statistics_for(sentences, name, tag_type) -> dict:
        if len(sentences) == 0:
            return {}

        classes_to_count = Corpus._count_sentence_labels(sentences)
        tags_to_count = Corpus._count_token_labels(sentences, tag_type)
        tokens_per_sentence = Corpus._get_tokens_per_sentence(sentences)

        label_size_dict = {}
        for l, c in classes_to_count.items():
            label_size_dict[l] = c

        tag_size_dict = {}
        for l, c in tags_to_count.items():
            tag_size_dict[l] = c

        return {
            "dataset": name,
            "total_number_of_documents": len(sentences),
            "number_of_documents_per_class": label_size_dict,
            "number_of_tokens_per_tag": tag_size_dict,
            "number_of_tokens": {
                "total": sum(tokens_per_sentence),
                "min": min(tokens_per_sentence),
                "max": max(tokens_per_sentence),
                "avg": sum(tokens_per_sentence) / len(sentences),
            },
        }

    @staticmethod
    def _get_tokens_per_sentence(sentences):
        return list(map(lambda x: len(x.tokens), sentences))

    @staticmethod
    def _count_sentence_labels(sentences):
        label_count = defaultdict(lambda: 0)
        for sent in sentences:
            for label in sent.labels:
                label_count[label.value] += 1
        return label_count

    @staticmethod
    def _count_token_labels(sentences, label_type):
        label_count = defaultdict(lambda: 0)
        for sent in sentences:
            for token in sent.tokens:
                if label_type in token.annotation_layers.keys():
                    label = token.get_tag(label_type)
                    label_count[label.value] += 1
        return label_count

    def __str__(self) -> str:
        return "Corpus: %d train + %d dev + %d test sentences" % (
            len(self.train) if self.train else 0,
            len(self.dev) if self.dev else 0,
            len(self.test) if self.test else 0,
        )

    def make_label_dictionary(self, label_type: str) -> Dictionary:
        """
        Creates a dictionary of all labels assigned to the sentences in the corpus.
        :return: dictionary of labels
        """
        label_dictionary: Dictionary = Dictionary(add_unk=True)
        label_dictionary.multi_label = False

        from flair.datasets import DataLoader

        datasets = [self.train]

        data = ConcatDataset(datasets)

        loader = DataLoader(data, batch_size=1)

        log.info("Computing label dictionary. Progress:")

        # if there are token labels of provided type, use these. Otherwise use sentence labels
        token_labels_exist = False

        all_label_types = Counter()
        all_sentence_labels = []
        for batch in Tqdm.tqdm(iter(loader)):

            for sentence in batch:

                # check for labels of words
                if isinstance(sentence, Sentence):
                    for token in sentence.tokens:
                        all_label_types.update(token.annotation_layers.keys())
                        for label in token.get_labels(label_type):
                            label_dictionary.add_item(label.value)
                            token_labels_exist = True

                # if we are looking for sentence-level labels
                if not token_labels_exist:
                    # check if sentence itself has labels
                    labels = sentence.get_labels(label_type)
                    all_label_types.update(sentence.annotation_layers.keys())

                    for label in labels:
                        if label.value not in all_sentence_labels: all_sentence_labels.append(label.value)

                    if not label_dictionary.multi_label:
                        if len(labels) > 1:
                            label_dictionary.multi_label = True

        # if this is not a token-level prediction problem, add sentence-level labels to dictionary
        if not token_labels_exist:
            for label in all_sentence_labels:
                label_dictionary.add_item(label)

        if len(label_dictionary.idx2item) == 0:
            log.error(
                f"Corpus contains only the labels: {', '.join([f'{label[0]} (#{label[1]})' for label in all_label_types.most_common()])}")
            log.error(f"You specified as label_type='{label_type}' which is not in this dataset!")

            raise Exception

        log.info(
            f"Corpus contains the labels: {', '.join([label[0] + f' (#{label[1]})' for label in all_label_types.most_common()])}")
        log.info(f"Created (for label '{label_type}') {label_dictionary}")

        return label_dictionary

    def get_label_distribution(self):
        class_to_count = defaultdict(lambda: 0)
        for sent in self.train:
            for label in sent.labels:
                class_to_count[label.value] += 1
        return class_to_count

    def get_all_sentences(self) -> Dataset:
        parts = []
        if self.train: parts.append(self.train)
        if self.dev: parts.append(self.dev)
        if self.test: parts.append(self.test)
        return ConcatDataset(parts)

    @deprecated(version="0.8", reason="Use 'make_label_dictionary' instead.")
    def make_tag_dictionary(self, tag_type: str) -> Dictionary:

        # Make the tag dictionary
        tag_dictionary: Dictionary = Dictionary(add_unk=False)
        tag_dictionary.add_item("O")
        for sentence in self.get_all_sentences():
            for token in sentence.tokens:
                tag_dictionary.add_item(token.get_tag(tag_type).value)
        tag_dictionary.add_item("<START>")
        tag_dictionary.add_item("<STOP>")
        return tag_dictionary


class MultiCorpus(Corpus):
    def __init__(self, corpora: List[Corpus], name: str = "multicorpus", **corpusargs):
        self.corpora: List[Corpus] = corpora

        train_parts = []
        dev_parts = []
        test_parts = []
        for corpus in self.corpora:
            if corpus.train: train_parts.append(corpus.train)
            if corpus.dev: dev_parts.append(corpus.dev)
            if corpus.test: test_parts.append(corpus.test)

        super(MultiCorpus, self).__init__(
            ConcatDataset(train_parts) if len(train_parts) > 0 else None,
            ConcatDataset(dev_parts) if len(dev_parts) > 0 else None,
            ConcatDataset(test_parts) if len(test_parts) > 0 else None,
            name=name,
            **corpusargs,
        )

    def __str__(self):
        output = f"MultiCorpus: " \
                 f"{len(self.train) if self.train else 0} train + " \
                 f"{len(self.dev) if self.dev else 0} dev + " \
                 f"{len(self.test) if self.test else 0} test sentences\n - "
        output += "\n - ".join([f'{type(corpus).__name__} {str(corpus)} - {corpus.name}' for corpus in self.corpora])
        return output


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag.value == "O":
            continue
        split = tag.value.split("-")
        if len(split) != 2 or split[0] not in ["I", "B"]:
            return False
        if split[0] == "B":
            continue
        elif i == 0 or tags[i - 1].value == "O":  # conversion IOB1 to IOB2
            tags[i].value = "B" + tag.value[1:]
        elif tags[i - 1].value[1:] == tag.value[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i].value = "B" + tag.value[1:]
    return True

# MODIFIED：target scheme.
def iob_i_only(tags):
    """
    IOB -> I
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.value == "O" or tag.value == "":
            new_tags.append("O")
        else:
            new_tags.append("I-" + tag.value.split("-")[1])
    return new_tags


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.value == "O" or tag.value == "":
            new_tags.append("O")
        elif tag.value.split("-")[0] == "B":
            if i + 1 != len(tags) and tags[i + 1].value.split("-")[0] == "I":
                new_tags.append(tag.value)
            else:
                new_tags.append(tag.value.replace("B-", "S-"))
        elif tag.value.split("-")[0] == "I":
            if i + 1 < len(tags) and tags[i + 1].value.split("-")[0] == "I":
                new_tags.append(tag.value)
            else:
                new_tags.append(tag.value.replace("I-", "E-"))
        else:
            raise Exception("Invalid IOB format!")
    return new_tags


def randomly_split_into_two_datasets(dataset, length_of_first):
    import random
    indices = [i for i in range(len(dataset))]
    random.shuffle(indices)

    first_dataset = indices[:length_of_first]
    second_dataset = indices[length_of_first:]
    first_dataset.sort()
    second_dataset.sort()

    return [Subset(dataset, first_dataset), Subset(dataset, second_dataset)]
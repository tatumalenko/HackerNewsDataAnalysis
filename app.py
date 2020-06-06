import csv
import re
import string
from collections import Counter
from math import log
from typing import List, Dict, Set


class Classification:
    text: str
    line: str
    class_type_predicted: str
    class_type_actual: str
    scores: List[float]
    accurate: bool

    def __init__(self, text: str, line: str, class_type_actual: str, class_type_predicted: str, scores: List[float]):
        self.text = text
        self.line = line
        self.class_type_actual = class_type_actual
        self.class_type_predicted = class_type_predicted
        self.scores = scores
        self.accurate = self.class_type_actual == self.class_type_predicted


class DataSet:
    _text_index: int
    _class_index: int
    lines: List[List[str]]
    removed_set: Set[str]
    class_types: Set[str]
    word_class_type_frequency_sums: Dict[str, int]
    vocabulary_size: int
    frequencies: Dict[str, Dict[str, int]]
    probabilities: Dict[str, Dict[str, float]]
    priors: Dict[str, float]

    def __init__(self, class_index: int, text_index: int, lines: List[List[str]]):
        self._text_index = text_index
        self._class_index = class_index
        self.lines = lines
        self.removed_set = set()
        self.class_types = set([line[self._class_index] for line in self.lines])
        self.frequencies = self._compute_frequencies()
        self.vocabulary_size = len(self.frequencies.keys())
        self.word_class_type_frequency_sums = self._compute_word_class_type_frequency_sums()
        self.probabilities = self._compute_probabilities()
        self.priors = self._compute_priors()

    def _compute_frequencies(self) -> Dict[str, Dict[str, int]]:
        frequencies = dict()
        class_type_frequency: Dict[str, int] = dict()
        for class_type in self.class_types:
            class_type_frequency[class_type] = 0

        for line in self.lines:
            sanitized_words = self._sanitize_text(line[self._text_index])
            for word in sanitized_words:
                frequencies[word] = class_type_frequency.copy()

        for line in self.lines:
            class_type = line[self._class_index]
            sanitized_words = self._sanitize_text(line[self._text_index])
            for word in sanitized_words:
                frequencies[word][class_type] += 1

        return frequencies

    def _compute_word_class_type_frequency_sums(self) -> Dict[str, int]:
        word_class_type_frequency_sums = dict()
        for class_type in self.class_types:
            word_class_type_frequency_sums[class_type] = sum(
                [self.frequencies[w][class_type] for w in self.frequencies.keys()])
        return word_class_type_frequency_sums

    def _compute_probabilities(self) -> Dict[str, Dict[str, float]]:
        probabilities = dict()
        class_type_probabilities: Dict[str, float] = dict()
        for class_type in self.class_types:
            class_type_probabilities[class_type] = 0.0

        for word in self.frequencies.keys():
            probabilities[word] = class_type_probabilities.copy()
            for class_type in self.class_types:
                probabilities[word][class_type] = self.probability(word, class_type)

        return probabilities

    def _compute_priors(self) -> Dict[str, float]:
        priors = dict(Counter([line[self._class_index] for line in self.lines]))
        for key in priors.keys():
            priors[key] /= sum(priors.values())
        return priors

    def _sanitize_text(self, text: str) -> List[str]:
        # ^\d+,\d+.*[, ]how .*,2018$
        # ^\d+,\d+.*[, ]how[, ].*,story,.*2018$
        # ,(?:https?://.*)?,\d+,\d+\.?\d+?,\d+\.?\d+?
        # set([w.lower() for w in re.split('\s', text)]).difference(sanitized)
        # return [word[0].lower() for word in
        #         filter(None, [re.findall(r'[\w-]+', word) for word in re.split(r'\s', text)])]
        sanitized = list(filter(None, [word.strip(string.punctuation).lower() for word in
                                       re.split(r'\s', re.sub(r'[^\w\'\s&+-.]+', '', text))]))
        removed = list(filter(None, set([w.lower() for w in re.split('', text)]).difference(str.join(' ', sanitized))))
        removed = str.join(' ', removed)
        removed = removed.strip(' ')
        if removed != '':
            self.removed_set.add(str.join(' ', removed))
        return sanitized

    def compute_classifications(self, csv_path: str):
        lines = [row for row in csv.reader(open(csv_path), delimiter=",")]
        headers, lines = list(filter(None, lines[0])), lines[1:]
        classifications = []
        for line in lines:
            classifications.append(self._compute_classification(line))
        return classifications

    def _compute_classification(self, line: List[str]) -> Classification:
        text = line[self._text_index]
        class_type_actual = line[self._class_index]
        scores, class_type_predicted = self.classify(text)
        return Classification(text, line, class_type_actual, class_type_predicted, scores)

    def probability(self, word: str, class_type: str) -> float:
        return (0.5 + self.frequencies[word][class_type]) / (
                self.word_class_type_frequency_sums[class_type] + self.vocabulary_size * 0.5)

    def score(self, class_type: str, sanitized: List[str]) -> float:
        score = log(self.priors[class_type], base=10)
        for word in sanitized:
            if word in self.probabilities.keys():
                score += log(self.probabilities[word][class_type], base=10)
        return score

    def classify(self, text: str) -> (List[float], str):
        sanitized = self._sanitize_text(text)
        scores = []
        argmax = float('-Inf')
        class_type_argmax = list(self.class_types)[0]
        for class_type in self.class_types:
            score = self.score(class_type, sanitized)
            scores.append(score)
            if score > argmax:
                argmax = score
                class_type_argmax = class_type
        return scores, class_type_argmax


class Main:
    _dataset_path: str
    _stopwords_path: str
    _text_index: int
    _class_index: int
    _year_index: int

    def __init__(self, dataset_path: str, stopwords_path: str, text_index: int, class_index: int, year_index: int):
        self._dataset_path = dataset_path
        self._stopwords_path = stopwords_path
        self._text_index = text_index
        self._class_index = class_index
        self._year_index = year_index

        lines = [row for row in csv.reader(open(self._dataset_path), delimiter=",")]
        headers, lines = list(filter(None, lines[0])), lines[1:]

        training_data = DataSet(text_index=self._text_index, class_index=self._class_index,
                                lines=[line for line in lines if line[year_index] == '2018'])
        testing_data = DataSet(text_index=self._text_index, class_index=self._class_index,
                               lines=[line for line in lines if line[year_index] == '2019'])

    def start(self):
        pass


if __name__ == '__main__':
    Main(
        dataset_path='./res/hns_2018_2019.csv',
        stopwords_path='./res/stopwords.txt',
        text_index=2,
        class_index=3,
        year_index=9).start()

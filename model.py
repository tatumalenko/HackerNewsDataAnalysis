import csv
import re
import string
from collections import Counter
from math import log10
from typing import Optional, List, Dict, Tuple, Set

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from nltk import PorterStemmer
from sklearn import metrics

from classification import Classification
from error import ArgumentError


class Model:
    _text_index: int
    _class_index: int
    _stop_words: Optional[List[str]]
    _min_max_size: Optional[Tuple[int, int]]
    _min_frequency: Optional[int]
    _top_percent_removed: Optional[float]
    _stemmer: PorterStemmer
    word_weights: Dict[str, float]
    smoothing: float
    lines: List[List[str]]
    removed_tokens: Set[str]
    removed_words: List[str]
    class_types: List[str]
    word_class_type_frequency_sums: Dict[str, int]
    vocabulary: List[str]
    vocabulary_size: int
    frequencies: Dict[str, Dict[str, int]]
    probabilities: Dict[str, Dict[str, float]]
    priors: Dict[str, float]

    def __init__(self,
                 text_index: int,
                 class_index: int,
                 word_weights: Dict[str, float],
                 smoothing: float,
                 lines: Optional[List[List[str]]] = None,
                 csv_path: Optional[str] = None,
                 stop_words: Optional[List[str]] = None,
                 min_max_size: Optional[Tuple[int, int]] = None,
                 min_frequency: Optional[int] = None,
                 top_percent_removed: Optional[float] = None):
        self._stemmer = PorterStemmer()
        self._stop_words = stop_words
        self._min_max_size = min_max_size
        self._min_frequency = min_frequency
        self._top_percent_removed = top_percent_removed
        self.word_weights = word_weights
        self.smoothing = smoothing
        if lines is None and csv_path is not None:
            lines = [row for row in csv.reader(open(csv_path), quotechar='"', delimiter=',')]
            headers, lines = list(filter(None, lines[0])), lines[1:]
        elif lines is None and csv_path is None:
            raise ArgumentError(None, 'Either `lines: List[List[str]]` or `csv_path: str` must be supplied')

        self._text_index = text_index
        self._class_index = class_index
        self.lines = lines
        self.removed_tokens = set()
        self.removed_words = []
        self.class_types = sorted(list(set([line[self._class_index] for line in self.lines])), reverse=True)
        self.frequencies = self._compute_frequencies()
        self.vocabulary = list(self.frequencies.keys())
        self.vocabulary_size = len(self.vocabulary)
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

        # Perform any vocabulary alterations if necessary
        vocabulary = list(frequencies.keys()).copy()
        if self._stop_words is not None:
            for stop_word in self._stop_words:
                if stop_word in frequencies.keys():
                    frequencies.pop(stop_word)
                    self.removed_words.append(stop_word)
        elif self._min_max_size is not None:
            min_size, max_size = self._min_max_size
            for word in vocabulary:
                if len(word) <= min_size or len(word) >= max_size:
                    frequencies.pop(word)
                    self.removed_words.append(word)
        elif self._min_frequency is not None:
            min_frequency = self._min_frequency
            for word in vocabulary:
                if sum(frequencies[word].values()) <= min_frequency:
                    frequencies.pop(word)
                    self.removed_words.append(word)
        elif self._top_percent_removed is not None:
            top_percent_removed = self._top_percent_removed
            idx_frequency_pairs = [(i, key, sum(frequencies[key].values())) for (i, key) in
                                   zip(range(len(frequencies.keys())), frequencies.keys())]
            idx_frequency_pairs_sorted = sorted(idx_frequency_pairs, key=lambda x: x[2], reverse=True)
            remove_idx = int(top_percent_removed / 100 * len(idx_frequency_pairs_sorted))
            remove_words = [key for (i, key, max_frequency) in idx_frequency_pairs_sorted[:remove_idx + 1]]
            for word in remove_words:
                frequencies.pop(word)
                self.removed_words.append(word)

        return frequencies

    def _compute_word_class_type_frequency_sums(self) -> Dict[str, int]:
        word_class_type_frequency_sums = dict()
        for class_type in self.class_types:
            word_class_type_frequency_sums[class_type] = sum(
                [self.frequencies[w][class_type] for w in self.vocabulary])
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
        total_samples = sum(priors.values())
        for key in priors.keys():
            priors[key] /= total_samples
        return priors

    def _sanitize_text(self, text: str) -> List[str]:
        # Basic removal of punctuation and split by space
        punctuation = re.sub(r'[$]', '', string.punctuation)  # remove all punctuation except $
        sanitized = list(filter(None, [word.strip(punctuation).lower() for word in
                                       re.split(r'\s', re.sub(r'[^\w\'\s&+-.$]+', '', text))]))

        # Find difference in what is used and what was given
        removed = list(filter(None, set([w.lower() for w in re.split('', text)]).difference(str.join(' ', sanitized))))
        removed = str.join(' ', removed)
        removed = removed.strip(' ')
        if removed != '':
            self.removed_tokens.add(str.join(' ', removed))

        # Perform some stemming to group related words together
        new = []
        for word in sanitized:
            new.append(self._stemmer.stem(word))
        sanitized = new

        # Strip surrounding punctuation (but keeping inner character punctuation)
        sanitized = [w.strip(string.punctuation) for w in sanitized]

        # Consider weighted word(s) as single vocabulary element
        joined = str.join(' ', sanitized)
        for weighted_word in self.word_weights.keys():
            if weighted_word in joined:
                cleaned = re.sub(weighted_word, '', joined)
                cleaned = re.split(' ', cleaned)
                cleaned.append(weighted_word)
                return list(filter(None, cleaned))

        return sanitized

    def _compute_classification(self,
                                line: List[str],
                                text_index: int,
                                class_index: int) -> Classification:
        text = line[text_index]
        class_type_actual = line[class_index]
        scores, class_type_predicted = self._classify(text)
        return Classification(text, line, class_type_actual, class_type_predicted, scores)

    def _classify(self, text: str) -> (Dict[str, float], str):
        sanitized = self._sanitize_text(text)
        scores = dict()
        argmax = float('-Inf')
        class_type_argmax = self.class_types[0]
        for class_type in self.class_types:
            score = self.score(class_type, sanitized)
            scores[class_type] = score
            if score > argmax:
                argmax = score
                class_type_argmax = class_type
        return scores, class_type_argmax

    def probability(self, word: str, class_type: str) -> float:
        return (self.smoothing + self.frequencies[word][class_type]) / (
                self.word_class_type_frequency_sums[class_type] + self.vocabulary_size * self.smoothing)

    def score(self, class_type: str, sanitized: List[str]) -> float:
        score = log10(self.priors[class_type])
        for word in sanitized:
            if word in self.probabilities.keys():
                multiplier = 1.0
                class_type_word = str.join(' ', class_type.split('_')).strip()
                if word == class_type_word:
                    multiplier = self.word_weights[word]
                score += log10(self.probabilities[word][class_type] * multiplier)
        return score

    def compute_classifications(self,
                                text_index: int,
                                class_index: int,
                                lines: Optional[List[List[str]]] = None,
                                csv_path: Optional[str] = None) -> List[Classification]:
        if lines is None and csv_path is not None:
            lines = [row for row in csv.reader(open(csv_path), delimiter=",")]
            headers, lines = list(filter(None, lines[0])), lines[1:]
        elif lines is None and csv_path is None:
            raise ArgumentError(None, 'Either `lines: List[List[str]]` or `csv_path: str` must be supplied')

        classifications = []
        for line in lines:
            classifications.append(self._compute_classification(line, text_index=text_index, class_index=class_index))

        return classifications

    def classify(self,
                 lines: Optional[List[List[str]]] = None,
                 beta: float = 1.0,
                 average: str = 'weighted',
                 csv_path: Optional[str] = None,
                 text_index: Optional[int] = None,
                 class_index: Optional[int] = None) -> 'Performance':
        classifications = \
            self.compute_classifications(lines=lines,
                                         csv_path=csv_path,
                                         text_index=text_index if text_index is not None else self._text_index,
                                         class_index=class_index if class_index is not None else self._class_index)
        return Model.Performance(self, classifications, beta=beta, average=average)

    class Performance:
        _class_types: List[str]
        beta: float
        average: str
        model: 'Model'
        classifications: List[Classification]
        confusion_matrix: np.ndarray
        accuracy: float
        precision: float
        recall: float
        f_measure: float
        matthews_corrcoef: float
        cohen_kappa_score: float
        jaccard_score: float

        def __init__(self,
                     model: 'Model',
                     classifications: List[Classification],
                     beta: Optional[float] = 1.0,
                     average: str = 'weighted'):
            self.beta = beta
            self.average = average
            self.model = model
            self.classifications = classifications
            self._class_types = self._compute_class_types()
            self.confusion_matrix, self.accuracy, self.precision, self.recall, self.f_measure, self.matthews_corrcoef, self.cohen_kappa_score, self.jaccard_score = self._compute_metrics()

        def __str__(self):
            labels = sorted(self._class_types, reverse=True)
            y_pred = [str.ljust(c.class_type_predicted, 8) for c in self.classifications if not c.accurate]
            y_true = [str.ljust(c.class_type_actual, 8) for c in self.classifications if not c.accurate]
            ys = [c.class_type_actual for c in self.classifications if not c.accurate]
            s = ''
            s += 'confusion_matrix:\n'
            s += f'{self._class_types}\n'
            s += f'{self.confusion_matrix}\n'
            s += f'accuracy: {self.accuracy}\n'
            s += f'precision: {self.precision}\n'
            s += f'recall: {self.recall}\n'
            s += f'f_measure: {self.f_measure}\n'
            s += f'mcc: {self.matthews_corrcoef}\n'
            s += f'ck: {self.cohen_kappa_score}\n'
            s += f'jaccard: {self.jaccard_score}\n'
            s += f'misclassifications: {len(y_true)}\n'
            s += str.join('\n',
                          [f'{class_type}: {len([y for y in ys if y == class_type])}' for class_type in labels]) + '\n'
            s += f'pred: {str.join("", y_pred)}\n'
            s += f'true: {str.join("", y_true)}\n'
            return s

        def _compute_class_types(self) -> List[str]:
            class_types = set()
            for classification in self.classifications:
                class_types.add(classification.class_type_actual)
                class_types.add(classification.class_type_predicted)
            return sorted(list(class_types), reverse=True)

        def _compute_metrics(self) -> Tuple[np.ndarray, float, float, float, float, float, float, float]:
            y_actual = [classification.class_type_actual for classification in self.classifications]
            y_predicted = [classification.class_type_predicted for classification in self.classifications]

            # Overall model metrics
            accuracy = metrics.accuracy_score(y_true=y_actual,
                                              y_pred=y_predicted)
            precision, recall, _, _ = metrics.precision_recall_fscore_support(y_actual,
                                                                              y_predicted,
                                                                              average=self.average,
                                                                              beta=self.beta)
            beta2 = self.beta ** 2
            f_meaure_denom = (beta2 * precision + recall)
            f_measure = (1 + beta2) * (precision * recall) / f_meaure_denom if f_meaure_denom > 0.0 else 0.0
            matthews_corrcoef = metrics.matthews_corrcoef(y_true=y_actual, y_pred=y_predicted)
            cohen_kappa_score = metrics.cohen_kappa_score(y_actual, y_predicted)
            jaccard_score = metrics.jaccard_score(y_true=y_actual, y_pred=y_predicted, average=self.average)

            # Class type specific metrics
            precisions = dict()
            recalls = dict()
            f_measures = dict()

            prfs = metrics.precision_recall_fscore_support(np.array(y_actual),
                                                           np.array(y_predicted),
                                                           labels=list(self._class_types),
                                                           beta=1.0)
            confusion_matrix = metrics.confusion_matrix(np.array(y_actual),
                                                        np.array(y_predicted),
                                                        labels=list(self._class_types))

            for i, class_type in enumerate(self._class_types):
                precisions[class_type] = prfs[0][i]
                recalls[class_type] = prfs[1][i]
                f_measures[class_type] = prfs[2][i]

            self._precisions = precisions
            self._recalls = recalls
            self._f_measures = f_measures

            return confusion_matrix, accuracy, precision, recall, f_measure, matthews_corrcoef, cohen_kappa_score, jaccard_score

        def plot(self, ax: Axes, x: int):
            ax.scatter([x], [self.accuracy], marker='*', facecolor='red', label='accuracy',
                       s=(plt.rcParams['lines.markersize'] + 1) ** 2)
            ax.scatter([x], [self.precision], marker='x', facecolor='blue', label='precision')
            ax.scatter([x], [self.recall], marker='2', facecolor='turquoise', label='recall',
                       s=(plt.rcParams['lines.markersize'] + 2) ** 2)
            ax.scatter([x], [self.f_measure], marker='+', facecolor='fuchsia', label='f_measure',
                       s=(plt.rcParams['lines.markersize'] + 1) ** 2)
            ax.scatter([x], [self.matthews_corrcoef], marker='D', facecolor='pink', label='matthews_corrcoef')
            ax.scatter([x], [self.cohen_kappa_score], marker='.', facecolor='black', label='cohen_kappa')
            ax.scatter([x], [self.jaccard_score], marker='1', facecolor='lime', label='jaccard',
                       s=(plt.rcParams['lines.markersize'] + 2) ** 2)

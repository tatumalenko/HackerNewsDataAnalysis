import csv
import re
import string
from collections import Counter
from math import log10
from typing import List, Dict, Set, Optional, Tuple, Union

import numpy as np
import sklearn.metrics as metrics
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from nltk.stem import PorterStemmer

matplotlib.use('MacOSX')
plt.rcParams.update({'figure.dpi': 350})
plt.rcParams.update({'font.size': 4})
plt.rcParams.update({'lines.markersize': 2})


class ArgumentError(Exception):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class Classification:
    text: str
    line: List[str]
    class_type_predicted: str
    class_type_actual: str
    scores: Dict[str, float]
    accurate: bool

    def __init__(self, text: str, line: List[str], class_type_actual: str, class_type_predicted: str,
                 scores: Dict[str, float]):
        self.text = text
        self.line = line
        self.class_type_actual = class_type_actual
        self.class_type_predicted = class_type_predicted
        self.scores = scores
        self.accurate = self.class_type_actual == self.class_type_predicted


class ModelPerformance:
    _class_types: List[str]
    beta: float
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

    def __init__(self, model: 'Model', classifications: List[Classification], beta: Optional[float] = 1.0):
        self.beta = beta
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
        precision, recall, f_measure, _ = metrics.precision_recall_fscore_support(y_actual,
                                                                                  y_predicted,
                                                                                  average='weighted',
                                                                                  beta=self.beta)
        matthews_corrcoef = metrics.matthews_corrcoef(y_true=y_actual, y_pred=y_predicted)
        cohen_kappa_score = metrics.cohen_kappa_score(y_actual, y_predicted)
        jaccard_score = metrics.jaccard_score(y_true=y_actual, y_pred=y_predicted, average='weighted')

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
        ax.scatter([x], [self.accuracy], marker='*', facecolor='red', label='accuracy')
        ax.scatter([x], [self.precision], marker='x', facecolor='blue', label='precision')
        ax.scatter([x], [self.recall], marker='2', facecolor='turquoise', label='recall')
        ax.scatter([x], [self.f_measure], marker='P', facecolor='purple', label='f_measure')
        ax.scatter([x], [self.matthews_corrcoef], marker='D', facecolor='pink', label='matthews_corrcoef')
        ax.scatter([x], [self.cohen_kappa_score], marker='.', facecolor='black', label='cohen_kappa')
        ax.scatter([x], [self.jaccard_score], marker='1', facecolor='lime', label='jaccard')


class Model:
    _text_index: int
    _class_index: int
    _stop_words: Optional[List[str]]
    _min_max_size: Optional[Tuple[int, int]]
    _min_frequency: Optional[int]
    _top_percent_removed: Optional[float]
    _stemmer: PorterStemmer
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
                 class_index: int,
                 text_index: int,
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
        # TODO: This line gives better results for lower freq # but slightly worse for baseline
        # sanitized = [w.strip("'") for w in sanitized]

        # Consider 'ask hn' and 'show hn' as one word to better predict type
        joined = str.join(' ', sanitized)
        if 'ask hn' in joined:
            cleaned = re.sub(r'ask hn', '', joined)
            cleaned = re.split(' ', cleaned)
            cleaned.append('ask hn')
            return list(filter(None, cleaned))
        elif 'show hn' in joined:
            cleaned = re.sub(r'show hn', '', joined)
            cleaned = re.split(' ', cleaned)
            cleaned.append('show hn')
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
        class_type_argmax = list(self.class_types)[0]
        for class_type in self.class_types:
            score = self.score(class_type, sanitized)
            scores[class_type] = score
            if score > argmax:
                argmax = score
                class_type_argmax = class_type
        return scores, class_type_argmax

    def probability(self, word: str, class_type: str) -> float:
        return (0.5 + self.frequencies[word][class_type]) / (
                self.word_class_type_frequency_sums[class_type] + self.vocabulary_size * 0.5)

    def score(self, class_type: str, sanitized: List[str]) -> float:
        score = log10(self.priors[class_type])
        for word in sanitized:
            if word in self.probabilities.keys():
                multiplier = 10000 if (word == 'ask hn' and class_type == 'ask_hn') or (
                        word == 'show hn' and class_type == 'show_hn') or (
                                              word == 'poll' and class_type == 'poll') else 1
                # multiplier = 1.0
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
                 csv_path: Optional[str] = None,
                 text_index: Optional[int] = None,
                 class_index: Optional[int] = None) -> ModelPerformance:
        classifications = \
            self.compute_classifications(lines=lines,
                                         csv_path=csv_path,
                                         text_index=text_index if text_index is not None else self._text_index,
                                         class_index=class_index if class_index is not None else self._class_index)
        return ModelPerformance(self, classifications)


class Main:
    _dataset_path: str
    _stopwords_path: str
    _text_index: int
    _class_index: int
    _year_index: int
    data: List[List[str]]
    data_train: List[List[str]]
    data_test: List[List[str]]
    model: Model
    baseline: ModelPerformance

    def __init__(self, dataset_path: str, stopwords_path: str, text_index: int, class_index: int, year_index: int):
        self._dataset_path = dataset_path
        self._stopwords_path = stopwords_path
        self._text_index = text_index
        self._class_index = class_index
        self._year_index = year_index

        self.data = self.read_lines(self._dataset_path)
        self.data_train = [line for line in self.data if line[year_index] == '2018']
        self.data_test = [line for line in self.data if line[year_index] == '2019']

        self.model, self.baseline = self.experiment_baseline(data_train=self.data_train, data_test=self.data_test)

    @staticmethod
    def read_lines(csv_path: str) -> List[List[str]]:
        lines = [row for row in csv.reader(open(csv_path), quotechar='"', delimiter=',')]
        lines = lines[1:]
        return lines

    @staticmethod
    def plot(ax: Axes, mps: List[ModelPerformance], labels: List[str], title: str):
        xs = range(1, len(mps) + 1)
        # xs = [mp.model.vocabulary_size for mp in mps]
        for x, mp in zip(xs, mps):
            mp.plot(ax=ax, x=x)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                   labels=(
                       'accuracy', 'precision', 'recall', 'f_measure', 'matthews_corrcoef', 'cohen_kappa', 'jaccard'))

        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_title(title)

    @staticmethod
    def save_model(model: Model, file_path: str):
        s = ''
        types = ['story', 'ask_hn', 'show_hn', 'poll']
        ctr = 1
        vocabulary = sorted(model.frequencies.keys())
        ctr_padding = 6
        word_padding = max([len(w) for w in vocabulary]) + 1
        frequency_padding = max([len(str(f)) for w in vocabulary for f in model.frequencies[w].values()]) + 1
        probability_padding = max([len(str(p)) for w in vocabulary for p in model.probabilities[w].values()]) + 1
        for w in vocabulary:
            s += f'{str.ljust(str(ctr), ctr_padding)}  {str.ljust(w, word_padding)}'
            for t in types:
                s += f'  {str.ljust(str(model.frequencies[w][t] if t in model.class_types else 0), frequency_padding)}  {str.ljust(str(model.probabilities[w][t] if t in model.class_types else 0.0), probability_padding)}'
            s += '\n'
            ctr += 1
        f = open(file_path, 'w')
        f.write(s)
        f.close()

    @staticmethod
    def save_classification(mp: ModelPerformance, file_path: str):
        s = ''
        types = ['story', 'ask_hn', 'show_hn', 'poll']
        ctr = 1
        ctr_padding = 6
        titles = [classification.text for classification in mp.classifications]
        title_padding = max([len(w) for w in titles]) + 1
        type_padding = max([len(w) for w in types]) + 1
        score_padding = max(
            [len(str(c.scores[t])) for c in mp.classifications for t in types if t in mp.model.class_types]) + 1
        outcome_padding = 6
        for c in mp.classifications:
            s += f'{str.ljust(str(ctr), ctr_padding)} {str.ljust(c.text, title_padding)} {str.ljust(c.class_type_predicted, type_padding)}'
            for t in types:
                s += f'{str.ljust(str(c.scores[t] if t in mp.model.class_types else 0), score_padding)}'
            s += f'{str.ljust(c.class_type_actual, type_padding)} {str.ljust("right" if c.accurate else "wrong", outcome_padding)}'
            s += '\n'
            ctr += 1
        s += f'\n{mp}\n'
        f = open(file_path, 'w')
        f.write(s)
        f.close()

    @staticmethod
    def save_words(words: Union[List[str], Set[str]], file_path: str):
        s = ''
        for word in words:
            s += f'{word}\n'
        f = open(file_path, 'w')
        f.write(s)
        f.close()

    def experiment_baseline(self,
                            data_train: List[List[str]],
                            data_test: List[List[str]]) -> Tuple[Model, ModelPerformance]:
        model = Model(text_index=self._text_index, class_index=self._class_index, lines=data_train)
        baseline = model.classify(data_test)
        self.save_model(model=model, file_path='model-2018.txt')
        self.save_words(words=model.removed_tokens, file_path='remove-words.txt')
        self.save_words(words=sorted(model.frequencies.keys()), file_path='vocabulary.txt')
        self.save_classification(mp=baseline, file_path='baseline-result.txt')
        print('Baseline:')
        print(baseline)
        print()
        return model, baseline

    def experiment_1(self, data_train: List[List[str]], data_test: List[List[str]]):
        stop_words = re.split('\n', open('./res/stopwords.txt').read())
        model_e1 = Model(text_index=self._text_index,
                         class_index=self._class_index,
                         lines=data_train,
                         stop_words=stop_words)
        performance_e1 = model_e1.classify(data_test)
        self.save_model(model=model_e1, file_path='stopword-model.txt')
        self.save_classification(mp=performance_e1, file_path='stopword-result.txt')
        print('Experiment 1 (stopwords):')
        print(performance_e1)
        print()

    def experiment_2(self, data_train: List[List[str]], data_test: List[List[str]]):
        model_e2 = Model(text_index=self._text_index,
                         class_index=self._class_index,
                         lines=data_train,
                         min_max_size=(2, 9))
        performance_e2 = model_e2.classify(data_test)
        self.save_model(model=model_e2, file_path='wordlength-model.txt')
        self.save_classification(mp=performance_e2, file_path='wordlength-result.txt')
        print('Experiment 2 (min_max_length=(2,9):')
        print(performance_e2)
        print()

    def experiment_3(self, data_train: List[List[str]], data_test: List[List[str]], title: str):
        e3_mf_1 = Model(text_index=self._text_index, class_index=self._class_index, lines=data_train,
                        min_frequency=1).classify(data_test)
        e3_mf_5 = Model(text_index=self._text_index, class_index=self._class_index, lines=data_train,
                        min_frequency=5).classify(data_test)
        e3_mf_10 = Model(text_index=self._text_index, class_index=self._class_index, lines=data_train,
                         min_frequency=10).classify(data_test)
        e3_mf_15 = Model(text_index=self._text_index, class_index=self._class_index, lines=data_train,
                         min_frequency=15).classify(data_test)
        e3_mf_20 = Model(text_index=self._text_index, class_index=self._class_index, lines=data_train,
                         min_frequency=20).classify(data_test)
        e3_tp_5 = Model(text_index=self._text_index, class_index=self._class_index, lines=data_train,
                        top_percent_removed=5).classify(data_test)
        e3_tp_10 = Model(text_index=self._text_index, class_index=self._class_index, lines=data_train,
                         top_percent_removed=10).classify(data_test)
        e3_tp_15 = Model(text_index=self._text_index, class_index=self._class_index, lines=data_train,
                         top_percent_removed=15).classify(data_test)
        e3_tp_20 = Model(text_index=self._text_index, class_index=self._class_index, lines=data_train,
                         top_percent_removed=20).classify(data_test)
        e3_tp_25 = Model(text_index=self._text_index, class_index=self._class_index, lines=data_train,
                         top_percent_removed=25).classify(data_test)

        e3_mf = [self.baseline, e3_mf_1, e3_mf_5, e3_mf_10, e3_mf_15, e3_mf_20]
        e3_tp = [self.baseline, e3_tp_5, e3_tp_10, e3_tp_15, e3_tp_20, e3_tp_25]

        for label, mp in zip(['ref', '5', '10', '15', '20', '25'], e3_tp):
            print(f'e3_tp_{label}:')
            print(mp)

        fig: Figure = plt.figure(figsize=(4, 2))
        fig.suptitle(title, fontsize=5)
        ax_e3_1: Axes
        ax_e3_2: Axes
        ax_e3_1, ax_e3_2 = fig.subplots(1, 2)
        self.plot(ax_e3_1, e3_mf, ['ref', '1', '5', '10', '15', '20'], title='Lower freq = x removed')
        self.plot(ax_e3_2, e3_tp, ['ref', '5', '10', '15', '20', '25'], title='Top freq = x% removed')

        plt.tight_layout()
        ax_e3_1.set_position((ax_e3_1.get_position().x0,
                              ax_e3_1.get_position().y0,
                              ax_e3_1.get_position().width,
                              ax_e3_1.get_position().height * 0.95))
        ax_e3_2.set_position((ax_e3_2.get_position().x0,
                              ax_e3_2.get_position().y0,
                              ax_e3_2.get_position().width,
                              ax_e3_2.get_position().height * 0.95))

        plt.show()

        print('|v|_mf: ' + str([m.model.vocabulary_size for m in e3_mf]))
        print('|v|_tp: ' + str([m.model.vocabulary_size for m in e3_tp]))

    def experiment_non_story(self, data_test: List[List[str]]):
        data_test_non_story = [line for line in data_test if line[self._class_index] != 'story']
        performance = self.model.classify(lines=data_test_non_story)
        print('Experiment Non-Story:')
        print(performance)
        pass

    def experiment_with_polls(self, data_train: List[List[str]], data_test: List[List[str]]):
        performance = Model(text_index=self._text_index,
                            class_index=self._class_index,
                            lines=data_train).classify(lines=data_test)
        print('Experiment with Equal Story, Ask HN, Show HN, and Polls:')
        print(performance)
        pass

    def prepare_balanced_data(self):
        lines = self.read_lines(csv_path='./res/hn.csv')
        stories = [line for line in lines if line[2] == 'story']
        ask_hns = [line for line in lines if line[2] == 'ask_hn']
        show_hns = [line for line in lines if line[2] == 'show_hn']
        polls = [line for line in lines if line[2] == 'poll']
        train = stories[:100] + ask_hns[:100] + show_hns[:100] + polls[:100]
        test = stories[101:202] + ask_hns[101:200] + show_hns[101:200] + polls[101:200]

        def save(data, path):
            f = open(path, 'w', newline='')
            writer = csv.writer(f, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                ['', 'Object ID', 'Title', 'Post Type', 'Author', 'Created At', 'URL', 'Points', 'Number of Comments'])
            ctr = 0
            for line in data:
                writer.writerow([str(ctr)] + line)
                ctr += 1
            f.close()

        save(train, 'train.csv')
        save(test, 'test.csv')

    def start(self):
        self.experiment_1(data_train=self.data_train, data_test=self.data_test)
        self.experiment_2(data_train=self.data_train, data_test=self.data_test)
        self.experiment_3(data_train=self.data_train, data_test=self.data_test, title='No Post Type Classification')
        self.experiment_3(data_train=self.read_lines('train.csv'),
                          data_test=self.read_lines('test.csv'),
                          title='All Post Type Classifications (Balanced)')
        # self.experiment_non_story(data_test=self.data_test)
        # self.experiment_with_polls(data_train=self.read_lines('train.csv'),
        #                            data_test=self.read_lines('test.csv'))
        # self.prepare_balanced_data()


if __name__ == '__main__':
    Main(
        dataset_path='./res/hns_2018_2019.csv',
        stopwords_path='./res/stopwords.txt',
        text_index=2,
        class_index=3,
        year_index=9).start()

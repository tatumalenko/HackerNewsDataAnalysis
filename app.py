import csv
import re
from pathlib import Path
from time import time
from typing import List, Set, Tuple, Union, Dict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing.io import TextIO

from model import Model
from settings import Settings


class App:
    _settings: Settings
    labels: List[str]
    word_weights: Dict[str, float]
    data: List[List[str]]
    data_train: List[List[str]]
    data_test: List[List[str]]
    model: Model
    baseline: Model.Performance

    def __init__(self, settings: Settings):
        self._settings = settings

        # Set plot related configurations for matplotlib and pyplot
        matplotlib.use(self._settings.plot.backend)
        plt.rcParams.update({'figure.dpi': self._settings.plot.dpi})
        plt.rcParams.update({'font.size': self._settings.plot.fontsize})
        plt.rcParams.update({'lines.markersize': self._settings.plot.markersize})
        plt.rcParams.update({'lines.linewidth': self._settings.plot.linewidth})

        # Split the input file data into training and test sets
        self.data = self.read_lines(self._settings.model.in_path_data)
        self.data_train = [line for line in self.data if
                           self.parse_year(
                               line[self._settings.model.index_created_at]) == self._settings.model.data_train_year]
        self.data_test = [line for line in self.data if
                          self.parse_year(
                              line[self._settings.model.index_created_at]) == self._settings.model.data_test_year]

        self.model, self.baseline = self.experiment_baseline(data_train=self.data_train, data_test=self.data_test)

    @staticmethod
    def parse_year(created_at: str) -> str:
        if len(created_at) < 4 or not (1000 <= int(created_at[:4]) <= 3000):
            raise Exception('parse_year', 'Created At column was less than 4 characters long or not a year format')
        return created_at[:4]

    @staticmethod
    def read_lines(csv_path: str) -> List[List[str]]:
        lines = [row for row in csv.reader(open(csv_path), quotechar='"', delimiter=',')]
        lines = lines[1:]
        return lines

    @staticmethod
    def plot(ax: Axes, mps: List[Model.Performance], labels: List[str], title: str):
        xs = list(reversed(range(1, len(mps) + 1)))
        xticklabels = [mp.model.vocabulary_size for mp in mps]
        for x, mp in zip(xs, mps):
            mp.plot(ax=ax, x=x)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                   labels=(
                       'accuracy', 'precision', 'recall', 'f_measure', 'matthews_corrcoef', 'cohen_kappa', 'jaccard'))

        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xticklabels])
        ax.set_xlabel('vocabulary words remaining')
        ax2 = ax.secondary_xaxis('top')
        ax2.set_xticks(xs)
        ax2.set_xticklabels(labels)
        ax2.set_xlabel(title)

    @staticmethod
    def open_and_create_dir_if_needed(file_path: str) -> TextIO:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return open(file_path, 'w')

    def save_model(self, model: Model, file_path: str):
        s = ''
        types = self._settings.model.labels
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
        f = self.open_and_create_dir_if_needed(file_path)
        f.write(s)
        f.close()

    def save_classification(self, mp: Model.Performance, file_path: str):
        s = ''
        types = self._settings.model.labels
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
                s += f'{str.ljust(str(c.scores[t] if t in mp.model.class_types else "-Inf"), score_padding)}'
            s += f'{str.ljust(c.class_type_actual, type_padding)} {str.ljust("right" if c.accurate else "wrong", outcome_padding)}'
            s += '\n'
            ctr += 1
        s += f'\n{mp}\n'
        f = self.open_and_create_dir_if_needed(file_path)
        f.write(s)
        f.close()

    def save_words(self, words: Union[List[str], Set[str]], file_path: str):
        s = ''
        for word in words:
            s += f'{word}\n'
        f = self.open_and_create_dir_if_needed(file_path)
        f.write(s)
        f.close()

    def experiment_baseline(self,
                            data_train: List[List[str]],
                            data_test: List[List[str]]) -> Tuple[Model, Model.Performance]:
        model = Model(self._settings.model.index_title,
                      self._settings.model.index_class,
                      self._settings.model.word_weights,
                      self._settings.model.smoothing,
                      data_train)
        baseline = model.classify(data_test, beta=self._settings.metric.beta, average=self._settings.metric.average)
        self.save_model(model=model, file_path=self._settings.model.out_path_model)
        self.save_words(words=model.removed_tokens, file_path=self._settings.model.out_path_remove_words)
        self.save_words(words=sorted(model.frequencies.keys()), file_path=self._settings.model.out_path_vocabulary)
        self.save_classification(mp=baseline, file_path=self._settings.metric.out_path_baseline)
        print('Baseline:')
        print(baseline)
        print()
        return model, baseline

    def experiment_1(self, data_train: List[List[str]], data_test: List[List[str]]):
        stop_words = re.split('\n', open(self._settings.model.in_path_stopwords).read())
        model = Model(self._settings.model.index_title,
                      self._settings.model.index_class,
                      self._settings.model.word_weights,
                      self._settings.model.smoothing,
                      data_train,
                      stop_words=stop_words)
        performance = model.classify(data_test, beta=self._settings.metric.beta, average=self._settings.metric.average)
        self.save_model(model=model, file_path=self._settings.model.out_path_stopword)
        self.save_classification(mp=performance, file_path=self._settings.metric.out_path_stopword)
        print('Experiment 1 (stopwords):')
        print(performance)

    def experiment_2(self, data_train: List[List[str]], data_test: List[List[str]]):
        size = self._settings.model.wordlength_min, self._settings.model.wordlength_max
        model = Model(self._settings.model.index_title,
                      self._settings.model.index_class,
                      self._settings.model.word_weights,
                      self._settings.model.smoothing,
                      data_train,
                      min_max_size=size)
        performance = model.classify(data_test, beta=self._settings.metric.beta, average=self._settings.metric.average)
        self.save_model(model=model, file_path=self._settings.model.out_path_wordlength)
        self.save_classification(mp=performance, file_path=self._settings.metric.out_path_wordlength)
        print(f'Experiment 2 (min_max_length=({size}):')
        print(performance)

    def experiment_3(self, data_train: List[List[str]], data_test: List[List[str]]):
        mfs = [1, 5, 10, 15, 20]
        tps = [5, 10, 15, 20, 25]
        e3_mf = [self.baseline] + [Model(self._settings.model.index_title,
                                         self._settings.model.index_class,
                                         self._settings.model.word_weights,
                                         self._settings.model.smoothing,
                                         data_train,
                                         min_frequency=mf).classify(data_test, beta=self._settings.metric.beta,
                                                                    average=self._settings.metric.average) for mf in
                                   mfs]
        e3_tp = [self.baseline] + [Model(self._settings.model.index_title,
                                         self._settings.model.index_class,
                                         self._settings.model.word_weights,
                                         self._settings.model.smoothing,
                                         data_train,
                                         top_percent_removed=tp).classify(data_test, beta=self._settings.metric.beta,
                                                                          average=self._settings.metric.average) for tp
                                   in tps]
        e3_mf_labels = ['ref'] + [str(mf) for mf in mfs]
        e3_tp_labels = ['ref'] + [str(tp) for tp in tps]
        for label, mf in zip(e3_mf_labels[1:], e3_mf[1:]):
            print(f'e3_mf_{label}:')
            print(mf)
        for label, tp in zip(e3_tp_labels[1:], e3_tp[1:]):
            print(f'e3_tp_{label}:')
            print(tp)

        fig: Figure = plt.figure(figsize=(4, 2))
        ax_e3_1: Axes
        ax_e3_2: Axes
        ax_e3_1, ax_e3_2 = fig.subplots(1, 2)
        self.plot(ax_e3_1, e3_mf, e3_mf_labels, title='lowest x frequency removed')
        self.plot(ax_e3_2, e3_tp, e3_tp_labels, title='top x frequency percentile removed')

        plt.tight_layout()

        print('|v|_mf: ' + str([m.model.vocabulary_size for m in e3_mf]))
        print('|v|_tp: ' + str([m.model.vocabulary_size for m in e3_tp]))

    def prepare_balanced_data(self):
        lines = self.read_lines(csv_path='./res/hn.csv')
        stories = [line for line in lines if line[2] == 'story']
        ask_hns = [line for line in lines if line[2] == 'ask_hn']
        show_hns = [line for line in lines if line[2] == 'show_hn']
        polls = [line for line in lines if line[2] == 'poll']
        train = stories[:25] + ask_hns[:25] + show_hns[:25] + polls[:25]
        test = stories[25:50] + ask_hns[25:50] + show_hns[25:50] + polls[25:50]

        def save(data, path, is_training_data):
            f = open(path, 'w', newline='')
            writer = csv.writer(f, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                ['', 'Object ID', 'Title', 'Post Type', 'Author', 'Created At', 'URL', 'Points', 'Number of Comments'])
            ctr = 0
            for line in data:
                line[4] = '2018' if is_training_data else '2019'
                writer.writerow([str(ctr)] + line)
                ctr += 1
            f.close()

        save(train, 'res/train2018.csv', True)
        save(test, 'res/test2019.csv', False)

    def start(self):
        start = time()
        self.experiment_1(data_train=self.data_train, data_test=self.data_test)
        # self.experiment_2(data_train=self.data_train, data_test=self.data_test)
        self.experiment_3(data_train=self.data_train, data_test=self.data_test)
        stop = time()
        print()
        print(f'Elapsed time: {(stop - start)} secs')
        plt.show()
        # self.prepare_balanced_data()


if __name__ == '__main__':
    # App(Settings('./settings.ini')).start()
    App(Settings('./test.ini')).start()

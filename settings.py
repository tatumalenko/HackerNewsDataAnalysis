from configparser import ConfigParser, ExtendedInterpolation
from typing import List, Dict

from error import ArgumentError


class ModelSettings:
    in_path_data: str
    out_path_model: str
    out_path_remove_words: str
    out_path_vocabulary: str
    out_path_wordlength: str
    in_path_stopwords: str
    out_path_stopword: str
    labels: List[str]
    word_weights: Dict[str, float]
    data_test_year: str
    data_train_year: str
    index_title: int
    index_class: int
    index_created_at: int
    wordlength_min: int
    wordlength_max: int
    smoothing: float

    def __init__(self,
                 in_path_data: str,
                 out_path_model: str,
                 out_path_remove_words: str,
                 out_path_vocabulary: str,
                 out_path_wordlength: str,
                 in_path_stopwords: str,
                 out_path_stopword: str,
                 labels: List[str],
                 word_weights: Dict[str, float],
                 data_test_year: str,
                 data_train_year: str,
                 index_title: int,
                 index_class: int,
                 index_created_at: int,
                 wordlength_min: int,
                 wordlength_max: int,
                 smoothing: float):
        self.in_path_data = in_path_data
        self.out_path_model = out_path_model
        self.out_path_remove_words = out_path_remove_words
        self.out_path_vocabulary = out_path_vocabulary
        self.out_path_wordlength = out_path_wordlength
        self.in_path_stopwords = in_path_stopwords
        self.out_path_stopword = out_path_stopword
        self.labels = labels
        self.word_weights = word_weights
        self.data_test_year = data_test_year
        self.data_train_year = data_train_year
        self.index_title = index_title
        self.index_class = index_class
        self.index_created_at = index_created_at
        self.wordlength_min = wordlength_min
        self.wordlength_max = wordlength_max
        self.smoothing = smoothing


class MetricSettings:
    average: str
    beta: float
    out_path_baseline: str
    out_path_stopword: str
    out_path_wordlength: str

    def __init__(self,
                 average: str,
                 beta: float,
                 out_path_baseline: str,
                 out_path_stopword: str,
                 out_path_wordlength: str):
        if average not in ['weighted', 'macro', 'micro']:
            raise ArgumentError(None, f'Metric.average must be one of {average}')
        self.average = average
        self.beta = beta
        self.out_path_baseline = out_path_baseline
        self.out_path_stopword = out_path_stopword
        self.out_path_wordlength = out_path_wordlength


class PlotSettings:
    dpi: int
    linewidth: int
    fontsize: int
    markersize: int
    backend: str

    def __init__(self,
                 dpi: int,
                 linewidth: int,
                 fontsize: int,
                 markersize: int,
                 backend: str):
        self.dpi = dpi
        self.linewidth = linewidth
        self.fontsize = fontsize
        self.markersize = markersize
        self.backend = backend


class Settings:
    _cfg: ConfigParser
    model: ModelSettings
    metric: MetricSettings
    plot: PlotSettings

    def __init__(self, ini_path: str):
        self._cfg = ConfigParser(interpolation=ExtendedInterpolation())
        self._cfg.read(ini_path)
        self.model = ModelSettings(self.get_str('Model', 'in_path_data'),
                                   self.get_str('Model', 'out_path_model'),
                                   self.get_str('Model', 'out_path_remove_words'),
                                   self.get_str('Model', 'out_path_vocabulary'),
                                   self.get_str('Model', 'out_path_wordlength'),
                                   self.get_str('Model', 'in_path_stopwords'),
                                   self.get_str('Model', 'out_path_stopword'),
                                   self.get_list('Model', 'labels'),
                                   self.get_dict('Model', 'word_weights'),
                                   self.get_str('Model', 'data_train_year'),
                                   self.get_str('Model', 'data_test_year'),
                                   self.get_int('Model', 'index_title'),
                                   self.get_int('Model', 'index_class'),
                                   self.get_int('Model', 'index_created_at'),
                                   self.get_int('Model', 'wordlength_min'),
                                   self.get_int('Model', 'wordlength_max'),
                                   self.get_float('Model', 'smoothing'))
        self.metric = MetricSettings(self.get_str('Metric', 'average'),
                                     self.get_float('Metric', 'beta'),
                                     self.get_str('Metric', 'out_path_baseline'),
                                     self.get_str('Metric', 'out_path_stopword'),
                                     self.get_str('Metric', 'out_path_wordlength'))
        self.plot = PlotSettings(self.get_int('Plot', 'dpi'),
                                 self.get_int('Plot', 'linewidth'),
                                 self.get_int('Plot', 'fontsize'),
                                 self.get_int('Plot', 'markersize'),
                                 self.get_str('Plot', 'backend'))

    def get_str(self, section: str, key: str) -> str:
        return self._cfg.get(section, key)

    def get_int(self, section: str, key: str) -> int:
        return self._cfg.getint(section, key)

    def get_float(self, section: str, key: str) -> float:
        return float(self.get_str(section, key))

    def get_list(self, section: str, key: str) -> List[str]:
        return [value for value in self.get_str(section, key).split('\n') if value != '']

    def get_dict(self, section: str, key: str) -> Dict[str, float]:
        kvs = dict()
        for value in self.get_str(section, key).split('\n'):
            if value != '':
                k, v = [v.strip() for v in value.split(',')]
                kvs[k] = float(v)
        return kvs

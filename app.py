import csv
import re
from typing import List, Dict, Set


class DataSet:
    _text_index: int
    _class_index: int
    lines: List[List[str]]
    class_types: Set[str]
    vocabulary: Dict[str, Dict[str, int]]

    def __init__(self, class_index: int, text_index: int, lines: List[List[str]]):
        self._text_index = text_index
        self._class_index = class_index
        self.lines = lines
        self.class_types = set([line[self._class_index] for line in self.lines])

        self.vocabulary = dict()

        class_type_frequency: Dict[str, int] = dict()
        for class_type in self.class_types:
            class_type_frequency[class_type] = 0

        for line in self.lines:
            sanitized_words = self._sanitize_text(line[self._text_index])
            for word in sanitized_words:
                self.vocabulary[word] = class_type_frequency.copy()

        for line in self.lines:
            class_type = line[self._class_index]
            sanitized_words = self._sanitize_text(line[self._text_index])
            for word in sanitized_words:
                self.vocabulary[word][class_type] += 1

    @staticmethod
    def _sanitize_text(text: str):
        # ^\d+,\d+.*[, ]how .*,2018$
        # ^\d+,\d+.*[, ]how[, ].*,story,.*2018$
        # ,(?:https?://.*)?,\d+,\d+\.?\d+?,\d+\.?\d+?
        return [word[0].lower() for word in filter(None, [re.findall(r'\w+', word) for word in re.split(r'\s', text)])]


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

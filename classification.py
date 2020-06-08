from typing import List, Dict


class Classification:
    text: str
    line: List[str]
    class_type_predicted: str
    class_type_actual: str
    scores: Dict[str, float]
    accurate: bool

    def __init__(self,
                 text: str,
                 line: List[str],
                 class_type_actual: str,
                 class_type_predicted: str,
                 scores: Dict[str, float]):
        self.text = text
        self.line = line
        self.class_type_actual = class_type_actual
        self.class_type_predicted = class_type_predicted
        self.scores = scores
        self.accurate = self.class_type_actual == self.class_type_predicted

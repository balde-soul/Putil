from enum import Enum


class Stage(Enum):
    Train=0
    TrainEvaluate=1
    Evaluate=2
    Test=3
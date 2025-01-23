import pandas as pd
from GenderPredictor import GenderPredictor


class GenderTest:
    def __init__(self, test_data_path: str, name_data_path) -> None:
        self.data_frame = pd.read_csv(test_data_path)
        self.gender_predictor = GenderPredictor(name_data_path)

    def test_sentence(self, sentence: str) -> str:
        
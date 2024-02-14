from iapytoo.train.predictions import Predictions


class Feature:
    def __init__(self, computer) -> None:
        self.actual = []
        self.predicted = []
        self.computer = computer

    def compute(self, predictions: Predictions, **kwargs):
        for i in range(predictions.actual.shape[0]):
            actual_del = self.computer(predictions.actual[i, 0, :], **kwargs)
            self.actual.append(actual_del)
            predicted_del = self.computer(predictions.predicted[i, 0, :], **kwargs)
            self.predicted.append(predicted_del)

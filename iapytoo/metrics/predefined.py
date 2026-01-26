import torch

from iapytoo.utils.config import Config
from iapytoo.metrics.metric import Metric
from iapytoo.predictions.predictors import Predictor


class AccuracyMetric(Metric):
    def __init__(self, config: Config, predictor: Predictor = None):
        super(AccuracyMetric, self).__init__(
            "accuracy", config, predictor=predictor)
        self.k = config.metrics.top_accuracy

    def compute(self):

        # Take care, for this metrics predicted and target do not have the same shape.
        # predicted : ouputs of the models - kind of probabilities
        # Target : label of the class
        _, top_pred = self.outputs.topk(self.k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(self.target.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        correct_k = correct[: self.k].reshape(-1).float().sum(0, keepdim=True)

        return {
            "top-1": torch.round(100.0 * correct_1 / self.target.shape[0], decimals=2),
            f"top-{self.k}": torch.round(
                100.0 * correct_k / self.target.shape[0], decimals=2
            ),
        }


class R2Metric(Metric):
    def __init__(self, config: Config, predictor: Predictor = None) -> None:
        super(R2Metric, self).__init__("r2", config, predictor=predictor)

    def compute(self):
        # for distributed
        self.gather()

        # Compute the mean of the target values
        target_mean = torch.mean(self.target, dim=0)

        # Compute the total sum of squares (SS_tot)
        ss_tot = torch.sum((self.target - target_mean) ** 2, dim=0)

        # Compute the residual sum of squares (SS_res)
        ss_res = torch.sum((self.target - self.predicted) ** 2, dim=0)

        # Compute the R² score
        r2_score = 1 - ss_res / ss_tot

        # Return the mean R² score across all output dimensions
        return {self.name: r2_score}


class MSMetric(Metric):
    def __init__(self, config: Config, predictor: Predictor = None) -> None:
        super(MSMetric, self).__init__(
            "mean_square", config, predictor=predictor)

    def _compute(self):
        diff = self.predicted - self.target
        return torch.mean(diff * diff)

    def compute(self):
        self.gather()
        ms = self._compute()
        self.results = {self.name: ms}
        return self.results


class RMSMetric(MSMetric):
    def __init__(self, config: Config, predictor: Predictor = None) -> None:
        Metric.__init__(self, "root_mean_square", config, predictor=predictor)
        self.name = "rms"

    def compute(self):
        self.gather()
        mean_squared_error = super()._compute()
        self.results = {self.name: torch.sqrt(mean_squared_error)}
        return self.results


class AccumulAccuracyMetric(Metric):
    def __init__(self, config: Config, predictor: Predictor = None):
        super(AccumulAccuracyMetric, self).__init__(
            "accumul_accuracy", config, predictor=predictor)
        self.k = config.metrics.top_accuracy
        self.correct_1 = 0
        self.correct_k = 0
        self.n = 0

    def update(self, outputs, target):
        # Take care, for this metrics predicted and target do not have the same shape.
        # predicted : ouputs of the models - kind of probabilities
        # Target : label of the class
        _, top_pred = outputs.topk(self.k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(target.view(1, -1).expand_as(top_pred))
        self.correct_1 += correct[:1].reshape(-1).float().sum(0, keepdim=True)
        self.correct_k += correct[: self.k].reshape(-1).float().sum(0, keepdim=True)
        self.n += target.shape[0]

    def compute(self):
        return {
            "top-1": torch.round(100.0 * self.correct_1 / self.n, decimals=2),
            f"top-{self.k}": torch.round(
                100.0 * self.correct_k / self.n, decimals=2
            ),
        }

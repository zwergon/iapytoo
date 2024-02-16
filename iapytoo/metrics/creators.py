from torchmetrics import R2Score


class MetricCreator(object):

    def __init__(self, name):
        self.name = name

    def create(self):
        pass


class R2Creator(MetricCreator):

    def __init__(self):
        super().__init__("r2")

    def create(self):
        return R2Score()

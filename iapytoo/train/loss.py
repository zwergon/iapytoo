from iapytoo.utils.iterative_mean import Mean


class Loss:
    def __init__(self) -> None:
        self.train_loss = None
        self.valid_loss = None

    def flush(self):
        self.train_loss.flush()
        self.valid_loss.flush()

    def reset(self):
        self.train_loss = Mean.create("ewm")
        self.valid_loss = Mean.create("ewm")

    def state_dict(self):
        return {
            "train": self.train_loss.state_dict(),
            "valid": self.valid_loss.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.train_loss.load_state_dict(state_dict=state_dict["train"])
        self.valid_loss.load_state_dict(state_dict=state_dict["valid"])

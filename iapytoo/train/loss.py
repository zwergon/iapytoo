from iapytoo.utils.iterative_mean import Mean


class Loss:
    def __init__(self, n_losses=2) -> None:
        self.n_losses = n_losses
        self.losses = []

    def flush(self):
        for loss in self.losses:
            loss.flush()

    def __call__(self, index):
        try:
            return self.losses[index]
        except IndexError:
            raise Exception(f"Index {index} of loss out of range")

    def reset(self):
        self.losses = [Mean.create("ewm") for _ in range(self.n_losses)]

    def state_dict(self):
        state_dict = {
            f"loss_{i}": self.losses[i].state_dict() for i in range(self.n_losses)
        }
        state_dict["n_losses"] = self.n_losses
        return state_dict

    def load_state_dict(self, state_dict):
        self.n_losses = state_dict["n_losses"]
        for i in range(self.n_losses):
            self.losses[i].load_state_dict(state_dict[f"loss_{i}"])

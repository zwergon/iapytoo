import unittest
from enum import Enum
from iapytoo.utils.iterative_mean import Mean
from iapytoo.train.loss import Loss


class LossType1(str, Enum):
    TRAIN = "train"
    VALID = "valid"


class LossType2(str, Enum):
    GEN = "generator"
    DISC = "discriminator"


class TestLoss(unittest.TestCase):

    def test_reset_and_call(self):
        loss = Loss(LossType1)
        loss.reset()

        # Vérifie que toutes les clés sont présentes
        self.assertIn(LossType1.TRAIN, loss.losses)
        self.assertIn(LossType1.VALID, loss.losses)

        # Vérifie que __call__ marche avec enum
        train_loss = loss(LossType1.TRAIN)
        self.assertIsInstance(train_loss, Mean)

        # Vérifie que __call__ marche aussi avec string
        valid_loss = loss("valid")
        self.assertIsInstance(valid_loss, Mean)

    def test_state_dict_and_load_state_dict(self):
        loss = Loss(LossType1)
        loss.reset()
        loss(LossType1.TRAIN).update(0.5)

        state = loss.state_dict()
        self.assertIn("train", state)
        self.assertIn("valid", state)

        # Recharge dans un autre objet
        new_loss = Loss(LossType1)
        new_loss.load_state_dict(state)

        self.assertAlmostEqual(
            new_loss(LossType1.TRAIN).value,
            loss(LossType1.TRAIN).value
        )

    def test_dynamic_enum(self):
        loss = Loss(LossType2)
        loss.reset()

        self.assertIn(LossType2.GEN, loss.losses)
        self.assertIn(LossType2.DISC, loss.losses)

        # Vérifie que state_dict reflète bien les clés
        state = loss.state_dict()
        self.assertIn("generator", state)
        self.assertIn("discriminator", state)


if __name__ == "__main__":
    unittest.main()

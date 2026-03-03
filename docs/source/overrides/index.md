# Overrides

Overrides are the core extension mechanism of iapytoo.

## The inference pipeline

When an MLflow model logged with iapytoo is served, the data flows through the following steps:

```
X (np.ndarray)
   │
   ▼
Transform (normalization, feature engineering, etc.)
   │
   ▼
Model.evaluate_one(...)   ← the model itself
   │
   ▼
Predictor(...)           ← post-processing / formatting
   │
   ▼
Y (np.ndarray)
```

Each block in this chain corresponds to an **attribute of `MlflowModelProvider`**.

## The `MlflowModelProvider` contract

At the heart of iapytoo overrides is the {py:class}`MlflowModelProvider <iapytoo.mlflow.model.MlflowModelProvider>` class. This abstract class defines **everything that is needed to package and serve a model with MLflow**.

A provider instance describes the *full inference chain* used when MLflow calls `predict` on a logged model.



---

### Provider attributes

A concrete provider subclass must define (explicitly or implicitly) the following attributes:

| Attribute        | Type        | Role in the pipeline |
|------------------|-------------|----------------------|
| `_transform`     | `Transform` | Pre-processes raw inputs before the model |
| `_model`         | `Model`     | The actual neural network / estimator |
| `_predictor`     | `Predictor` | Converts raw model outputs into final `Y` |
| `_input_example` | `np.array`  | Optional example input for MLflow |

Some of these have **defaults**:

* `_predictor` defaults to a generic `Predictor()`
* `_transform` and `_input_example` can be left as `None`

The only mandatory part is the **model itself** (`_model`).

---

### How `predict` is executed

When MLflow calls `predict`, iapytoo executes the following logic (simplified from the real code):

1. Load the input into a NumPy batch `X`
2. Apply the optional transform:
   ```python
   if self.transform is not None:
       X = self.transform(X)
   ```
3. Convert to a torch tensor and run the model:
   ```python
   outputs = self.model.evaluate_one(X_tensor)
   ```
4. Apply the predictor:
   ```python
   Y = self.ml_predictor(outputs)
   ```

This is why the provider defines the **full inference chain**, not just the model.

---

### Minimal provider example

Here is a minimal override of `MlflowModelProvider`:

```python
class MyProvider(MlflowModelProvider):
    def __init__(self, config):
        super().__init__(config)
        self._model = MyTorchModel(config)
        self._transform = MyTransform(config)
        self._predictor = MyPredictor()

    def code_definition(self):
        return {
            "path": str(Path(__file__).parent),
            "provider": {
                "module": "myproject.provider",
                "class": "MyProvider"
            }
        }
```

---

### Mental model

Think of `MlflowModelProvider` as the object that **glues together**:

* data preprocessing
* the trained model
* post-processing
* and MLflow packaging

➡️ If you control the provider, you control the *entire* prediction semantics of your model.



```{toctree}
:maxdepth: 1

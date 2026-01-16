MNIST Classification with iapytoo and MLflow
=========================================

This page documents how to train, log, and deploy a **MNIST classification** model using
``iapytoo`` on top of **MLflow**, and how to perform inference both from Python
(``mlflow.pyfunc``) and via ``mlflow models serve``.

The goal is to clearly explain:

- Which pieces of code must be **overridden / customized** ("overloads")
- How the **training** is launched
- How to perform **Python inference** using ``pyfunc``
- How to serve the model with **MLflow Models Serve**
- How to **define and document the predict signature**

This documentation is based on the following example files:

- Training script: ``examples/mnist.py``
- Configuration file: ``examples/config_mnist.yml``
- Custom overrides directory: ``examples/examples/``


Project structure
-----------------

A typical MNIST example using ``iapytoo`` is organized as follows::

    iapytoo/
    ├── examples/
    │   ├── mnist.py
    │   ├── config_mnist.yml
    │   └── examples/
    │       ├── dataset.py
    │       ├── model.py
    │       ├── trainer.py
    │       └── inference.py

The ``examples/examples`` directory contains **project-specific overrides**.
These Python files are dynamically imported by ``iapytoo`` based on the configuration.


Concept: override-based customization
-------------------------------------

``iapytoo`` relies on a **convention-over-configuration** approach.

Instead of modifying the framework code, you provide **custom implementations** for:

- Dataset loading and preprocessing
- Model definition
- Training loop
- Inference logic

These implementations are referenced in the YAML configuration file and loaded at runtime.


Configuration file (config_mnist.yml)
-------------------------------------

The configuration file defines:

- The experiment name
- The MLflow tracking URI
- Which Python overrides are used
- Training hyperparameters

Example (simplified)::

    experiment:
      name: mnist_classification

    dataset:
      module: examples.dataset
      class: MnistDataset

    model:
      module: examples.model
      class: MnistModel

    trainer:
      module: examples.trainer
      class: MnistTrainer

    inference:
      module: examples.inference
      class: MnistPyfuncModel

    training:
      batch_size: 64
      epochs: 10
      learning_rate: 0.001

Each ``module`` is resolved relative to ``examples/examples``.


Dataset override
----------------

The dataset override is responsible for:

- Downloading or loading MNIST
- Applying transformations
- Returning data in a framework-compatible format

Typical responsibilities:

- ``__init__``: configuration handling
- ``get_train_dataloader``
- ``get_val_dataloader``
- ``get_test_dataloader``

This class is used **only during training**, not during inference.


Model override
--------------

The model override defines the neural network architecture.

Responsibilities:

- Build the network (e.g. PyTorch / TensorFlow)
- Expose forward / call method

Example responsibilities:

- Define convolutional layers
- Define classifier head
- Handle device placement

The model object is logged to MLflow during training.


Trainer override
----------------

The trainer override encapsulates the training loop.

Typical responsibilities:

- Optimizer and loss definition
- Epoch and batch loops
- Logging metrics to MLflow
- Saving checkpoints / final model

``iapytoo`` calls the trainer with:

- The instantiated dataset
- The instantiated model
- The training configuration


Launching the training
----------------------

Training is launched by running the example script::

    python examples/mnist.py --config examples/config_mnist.yml

The script typically performs:

1. Configuration loading
2. Dynamic import of overrides
3. MLflow experiment setup
4. Training execution
5. Model logging with ``mlflow.pyfunc``

At the end of training, an MLflow model artifact is created.


Inference override (iapytoo PyFunc model)
-----------------------------------

In ``iapytoo``, the PyFunc layer is **not a thin wrapper** around a trained model.
It is a **full orchestration layer** implemented in ``mlflow_model.py`` and built around
four core concepts:

- Dynamic code loading
- Configuration rehydration
- Valuator / Predictor separation
- Optional inference-specific transform and predictor

The MLflow model that is logged is always an instance of ``iapytoo.mlflow_model.MlflowModel``.


Key classes and responsibilities
--------------------------------

``MlflowModel``
^^^^^^^^^^^^^^^

This class inherits from ``mlflow.pyfunc.PythonModel`` and defines the **runtime contract**
used both by ``mlflow.pyfunc.load_model`` and ``mlflow models serve``.

Its responsibilities are:

- Load all artifacts at serving time (model, config, code)
- Rebuild the training-time objects (model, valuator, predictors)
- Expose a stable ``predict`` API


``MlflowModel.load_context``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the model is loaded, MLflow calls ``load_context``.

``iapytoo`` uses this hook to:

1. Read a ``code_definition.yml`` artifact
2. Add a zipped Python package to ``sys.path``
3. Reload the original YAML configuration
4. Re-instantiate:
   - the model class
   - the valuator
   - the predictor(s)
   - an optional inference transform

All this logic is implemented in the private helper:

- ``_MlflowModelPrivate.from_context``

This ensures **strict parity between training and inference**.


Model provider abstraction
--------------------------

``MlflowModelProvider`` is an optional abstraction used **at logging time**.

Its role is to declare:

- Which Python code must be shipped with the model
- Which model class must be reloaded
- Which optional transform is required

It provides:

- ``code_definition()``: a declarative description of runtime code
- ``code_path``: path to the Python sources to zip
- ``input_example``: optional NumPy example used by MLflow

The resulting ``code_definition.yml`` is stored as an MLflow artifact and drives
runtime imports.


Input contract for predict
--------------------------

Unlike standard MLflow PyFunc models, ``iapytoo`` defines a **very explicit input contract**.

``predict`` signature::

    predict(
        context: PythonModelContext,
        model_input: list[str | numpy.ndarray],
        params: dict | None
    )

Rules:

- ``model_input`` **must be a list or tuple**
- Each element is either:
  - a ``.npy`` file path
  - a ``numpy.ndarray``

Special value:

- ``"input_example"`` loads the stored MLflow input example

This design allows:

- File-based inference
- Batch inference
- Compatibility with MLflow serving JSON payloads


Inference pipeline
------------------

At runtime, inference follows these steps:

1. Load NumPy arrays from paths or in-memory arrays
2. Stack arrays into a batch: ``(N, ...)``
3. Apply optional ``MlflowTransform``
4. Convert batch to a Torch tensor
5. Run ``valuator.evaluate_one``
6. Post-process outputs using a ``Predictor``

This separation makes it possible to:

- Share evaluators between training and inference
- Use a **different predictor for ML inference**


Predictors and inference specialization
---------------------------------------

``iapytoo`` supports **inference-specific predictors**.

At training time, the following keys are stored in MLflow metadata:

- ``valuator_key``
- ``predictor_key``
- ``inference_key`` (optional)
- ``inference_args`` (optional)

At inference time:

- If ``inference_key`` is defined, it overrides the training predictor
- Otherwise, the training predictor is reused

This allows, for example:

- Argmax during training
- Probability vectors during inference


Return value
------------

The return value of ``predict`` is the **direct output of the predictor**.

Typical outputs:

- ``numpy.ndarray``
- ``list``
- JSON-serializable structures

No Pandas DataFrame is required, which makes the model compatible with
high-performance and file-based workflows.


Example PyFunc predict implementation
-------------------------------------

-------------------------------------

The ``predict`` method must:

1. Accept a ``pandas.DataFrame``
2. Convert inputs to tensors
3. Run the model in evaluation mode
4. Return a ``pandas.DataFrame``

Important constraints:

- No training-only logic
- No dependency on the original dataset class
- Deterministic output


Python inference using mlflow.pyfunc
------------------------------------

Once the model is logged, it can be loaded in Python::

    import mlflow.pyfunc

    model = mlflow.pyfunc.load_model("models:/mnist_classification/Production")

    predictions = model.predict(df)

Where ``df`` is a ``pandas.DataFrame`` matching the predict signature.

This is the recommended approach for:

- Batch inference
- Offline evaluation
- Integration in Python services


Inference with mlflow models serve
---------------------------------

The same PyFunc model can be served as an HTTP service::

    mlflow models serve \
        --model-uri models:/mnist_classification/Production \
        --env-manager local \
        --port 5000

The server exposes a REST API compatible with MLflow.


Defining an entry point
----------------------

The entry point is automatically defined by MLflow when logging a PyFunc model.

Key points:

- ``python_model``: your inference override class
- ``artifacts``: paths to model weights
- ``conda.yaml`` or ``requirements.txt``: runtime dependencies

Make sure that:

- ``iapytoo`` is available in the serving environment
- The override module is importable


Summary
-------

This MNIST example demonstrates the full lifecycle:

- Override-based customization
- Reproducible training with MLflow
- Clear and stable inference contract
- Multiple inference modes (Python / REST)

The same structure can be reused for more complex classification or regression tasks.


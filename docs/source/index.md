# IaPyToo

*IaPyToo* is a lightweight framework built on top of **MLflow** to standardize
training, logging, and deployment of machine learning models using an
**override-based architecture**.

The core idea is to:

* Keep training code flexible and explicit
* Preserve strict parity between training and inference
* Deploy models as reproducible, self-contained MLflow artifacts

This documentation is organized around four main concepts:

* **Overrides**: how users plug their own code into iapytoo
* **Training**: how experiments are configured and executed
* **Inference**: how models are loaded, served, and queried
* **Examples**: end-to-end projects to learn iapytoo by practice


## Quick overview
If you prefer learning by doing, check the new **Examples** section. It contains full projects, including:

* A **Deep Learning classification** example on the MNIST dataset
* A **Wasserstein GAN** that learns to generate a 1D sinusoidal curve

These examples are another way to discover iapytoo through concrete, runnable use cases.


An iapytoo project typically looks like this::

```
project/
├── config.yml
├── train.py
└── mlcode/
    ├── __init__.py
    ├── dataset.py
    ├── model.py
    ├── trainer.py
    └── provider.py
```

* Training is driven by a YAML configuration file
* Custom code lives in override modules (*mlcode* directory)
    * key module is given by overriding the {py:class}`MlflowModelProvider <iapytoo.train.mlflow_model.MlflowModelProvider>` class

* Models are logged as MLflow PyFunc models
* Inference reuses the exact same code and configuration

## Design principles

* **Explicit over implicit**: no hidden magic, all runtime behavior is declared
* **Reproducibility first**: everything required for inference is logged
* **Training / inference symmetry**: same model, same evaluators, same predictors
* **MLflow-native**: compatible with `mlflow.pyfunc` and `mlflow models serve`

## Getting started

If you are new to iapytoo, we recommend reading the documentation in this order:

1. :doc:`overrides/index` – how to write project-specific code
2. :doc:`training/index` – how to configure and run experiments
3. :doc:`inference/index` – how to deploy and query trained models
4. :doc:`examples/index` – learn by example with complete projects



```{toctree}
:maxdepth: 2
:caption: User Guide

overrides/index
training/index
inference/index
examples/index
api/index.rst

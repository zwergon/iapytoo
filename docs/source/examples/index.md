# Examples 

## Learn iapytoo by Practice

This section is for users who prefer to learn by doing. Each example is a **complete, runnable project** that showcases how iapytoo structures training, logging, and inference with MLflow.

The goals of the Examples section are to:

* Provide end-to-end reference projects
* Show how overrides, training, and inference fit together
* Demonstrate both classical deep learning and generative modeling

```{toctree}
:maxdepth: 1

mnist_example
wgan_example
```

---

## 1️⃣ MNIST — Deep Learning Classification

**Category:** Supervised learning / Image classification

### What you will learn

This example shows how to train a classical deep learning model on the MNIST handwritten digits dataset using iapytoo inside the shared **`examples` project**. You will see how to:

* Define a dataset + model override under `examples/mnist/`
* Implement a training script (`mnist_train.py`) driven by `config_mnist.yml`
* Log the trained model as an MLflow PyFunc artifact
* Reload the model for inference with `mnist_infer.py` using `config_infer.yml`. This script illustrates how to use
`iapytoo` to use a generated model during training in `mlflow pyfunc` format for logging inference on `dataloader` dataset using `mlflow`.
* Reload the model for inference with `mlflow_infer.py` using the `mlflow pyfunc` model in a pythonic way as does the 
endpoint in `mlflow models serve`


### Why this example matters

MNIST is a "hello world" of deep learning. This project demonstrates how iapytoo keeps **training and inference perfectly aligned** while remaining explicit and reproducible — even when multiple examples live in the same repository.

### Typical project structure (MNIST part)

```
examples_project/
├── config_mnist.yml
├── mnist_train.py
├── mnist_infer.py
├── mlflow_infer.py
└── examples/
    └── mnist/
        ├── model.py
        ├── provider.py
        ├── scheduler.py
        └── __init__.py
```

---


## 2️⃣ WGAN — Learning a 1D Sinusoidal Distribution

**Category:** Generative modeling / Wasserstein GAN

### What you will learn

This example shows how to train a **Wasserstein GAN (WGAN)** to generate samples from a simple 1D sinusoidal curve, also inside the shared **`examples` project**. You will see how to:

* Define a synthetic dataset sampling a sine curve (`examples/wgan/dataset.py`)
* Implement a Generator and a Critic (`generator.py`, `critic.py`)
* Train a WGAN with a custom loop in `wgan_train.py` driven by `config_wgan.yml`
* Log the generator (and full pipeline) with MLflow
* Serve the trained generator for inference using `wgan_infer.py`

### Why this example matters

This project demonstrates that iapytoo is not limited to classical supervised learning. It shows how to:

* Handle **non-standard training loops**
* Log complex generative models
* Preserve reproducibility even for adversarial setups — within the same examples repository

### Typical project structure (WGAN part)

```
examples_project/
├── config_wgan.yml
├── wgan_train.py
├── wgan_infer.py
└── examples/
    └── wgan/
        ├── dataset.py
        ├── generator.py
        ├── critic.py
        ├── dft_layer.py
        ├── provider.py
        └── __init__.py
```

---

Each example page will walk you through:

* The relevant configuration file (`config_mnist.yml` or `config_wgan.yml`)
* The override classes under `examples/`
* The training command (`*_train.py`)
* The resulting MLflow artifacts
* How to run inference with the trained model (`*_infer.py`)

➡️ Start with the MNIST example if you are new to iapytoo, then move on to the WGAN example to explore more advanced use cases.


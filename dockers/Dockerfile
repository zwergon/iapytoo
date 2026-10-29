# Image GPU pour faire tourner iapytoo (et les projets qui en dépendent),
# basée sur l'image CUDA publique de Docker Hub. Le contexte de build est ce
# dossier :
#
#   IAPYTOO_REF=v0.1.2 docker compose build      # -> iapytoo:v0.1.2
#   (ou : docker build --build-arg IAPYTOO_REF=v0.1.2 -t iapytoo:v0.1.2 .)
#
# iapytoo est installé depuis GitHub (ARG IAPYTOO_REF, branche develop par
# défaut), pas depuis la copie locale : le contexte reste ce dossier.

FROM nvidia/cuda:12.6.3-cudnn-runtime-rockylinux9

# Proxy éventuel : dnf et pip lisent http_proxy / https_proxy / no_proxy.
# Docker les transmet aux RUN sans ARG ni trace dans l'image s'ils sont
# passés en --build-arg (docker-compose.yml) ou dans ~/.docker/config.json
# ("proxies").

# Python 3.11 système (Rocky 9 : 3.9 par défaut) ; git pour installer
# iapytoo depuis GitHub.
RUN dnf install -y python3.11 git \
    && dnf clean all

# torch d'abord, depuis l'index PyTorch, en cu126 (CUDA de l'image de base) :
# 2.9.1 = borne haute de pyproject.toml (torch<=2.9.1), pour que
# l'installation d'iapytoo ne le remplace pas.
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
ARG IAPYTOO_REF=develop

COPY requirements.txt /tmp/requirements.txt
RUN python3.11 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir \
    torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
    --index-url https://download.pytorch.org/whl/cu126 \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    "iapytoo[mlserver] @ git+https://github.com/zwergon/iapytoo.git@${IAPYTOO_REF}" \
    && rm /tmp/requirements.txt

ENV PATH=/opt/venv/bin:${PATH} \
    MPLCONFIGDIR=/tmp/matplotlib \
    MPLBACKEND=Agg

CMD ["python", "-c", "import importlib.metadata as m, torch; print('iapytoo', m.version('iapytoo'), '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"]

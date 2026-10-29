
# Création de l'image Docker

Image GPU (CUDA 12.6, Rocky Linux 9, Python 3.11) avec torch 2.9.1 et
`iapytoo[mlserver]` installé depuis GitHub, dans le venv `/opt/venv`.

Dans ce répertoire `dockers/` :
```
docker compose build                          # -> iapytoo:develop
IAPYTOO_REF=v0.1.2 docker compose build       # -> iapytoo:v0.1.2 (tag, branche ou sha ; pas de "/")
```
ou sans compose :
```
docker build --build-arg IAPYTOO_REF=develop -t iapytoo:develop .
```

Variables lues dans l'environnement ou dans `dockers/.env` (non versionné) :
`IAPYTOO_REF`, `http_proxy` / `https_proxy` / `no_proxy` (passés en build
args, non conservés dans l'image) et `DOCKERFILE` (défaut `Dockerfile`, image
CUDA de Docker Hub ; un Dockerfile local non versionné, par exemple basé sur
un registre interne, peut être choisi ainsi).

Lancer un exemple avec le code local monté (prioritaire sur l'iapytoo
installé dans l'image, via `PYTHONPATH`), depuis la racine du dépôt :
```
dockers/run-python.sh
```

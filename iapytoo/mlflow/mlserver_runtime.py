"""
Runtime mlserver pret a l'emploi pour un modele iapytoo/MLflow, a utiliser dans
model_settings.json ("implementation": "iapytoo.mlflow.mlserver_runtime.MLflowRuntime")
a la place de "mlserver_mlflow.MLflowRuntime" directement.

Pourquoi ce fichier existe : mlserver_mlflow.MLflowRuntime.predict() fait
`decoded_payload = self.decode_request(payload)` SANS default_codec -- la
resolution du codec (mlserver.codecs.utils.decode_inference_request) se fait
uniquement via une recherche par content_type dans le registre global peuple
par @register_input_codec/@register_request_codec (iapytoo.mlflow.codec :
MlInputCodec/MlRequestCodec, content_type "mlmodelinput"). Ces decorateurs ne
s'executent qu'a l'IMPORT de ce module -- et rien ne l'importe cote serveur :
ni iapytoo/mlflow/__init__.py (vide), ni le chargement standard d'un modele
MLflow (mlflow.pyfunc.load_model -> MlflowModel.from_context, qui n'importe
que le module du PROVIDER, jamais iapytoo.mlflow.codec), ni mlserver_mlflow
lui-meme (qui ne connait rien de ce codec specifique a iapytoo), ni un
entry_point mlserver declare par iapytoo (aucun n'existe). Sans cet import,
decode_request ne trouve aucun codec pour content_type="mlmodelinput" et
retombe silencieusement sur la InferenceRequest brute, que mlflow rejette
ensuite avec "Expected list, but got InferenceRequest" (sa validation de type
hint attend list[MlInput]) -- verifie empiriquement (cf. CHANGELOG). Ce
probleme est independant du conditionnement (MlInput.condition) : il se
produirait deja pour un MlInput sans condition, des que content_type=
"mlmodelinput" est utilise dans un vrai deploiement mlserver (par opposition
aux scripts de test client qui, eux, importent iapytoo.mlflow.codec pour
ENCODER la requete, ce qui n'aide pas le DEcodage cote serveur).

Cette sous-classe ne fait qu'importer le module codec comme effet de bord
avant de deleguer entierement a mlserver_mlflow.MLflowRuntime -- aucune
surcharge de comportement.

Necessite l'extra "mlserver" (pip install "iapytoo[mlserver]") -- mlserver ne
s'installe pas sous Windows, ce module n'est donc utilisable que dans le
process mlserver lui-meme (Linux/WSL), jamais depuis les scripts train/infer.
"""
from iapytoo.mlflow.codec import MlInputCodec, MlRequestCodec  # noqa: F401 -- registers codecs
from mlserver_mlflow import MLflowRuntime as _MLflowRuntime


class MLflowRuntime(_MLflowRuntime):
    pass

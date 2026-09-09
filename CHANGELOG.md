# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- CI (`python-app.yml`) `build` job silently stopped installing the project's runtime
  dependencies (`torch`, `numpy`, `mlflow`, ...) once `requirements.txt` was removed in favor
  of `pyproject.toml` (commit `94754d5`): the step only ran `pip install -r requirements.txt`
  when that file existed, so it became a no-op. Every test module failed to import
  (`ModuleNotFoundError: No module named 'torch'`/`'numpy'`), which failed `build` and, since
  `publish` depends on it (`needs: build`), silently skipped the PyPI upload — `v0.1.1` was
  tagged and merged but never actually published. Fixed by installing the package itself
  (`pip install -e .`) instead of the dead `requirements.txt` path.

## [0.1.1] - 2026-09-09

Changes below were made in the sibling `ifpen-wind-diffusion` project while adapting it to
`iapytoo`, for gaps that turned out to be generic (architecture-agnostic) rather than specific
to that project — not yet committed.

### Added

- `Factory.register_backbone(key, backbone_cls)` / `Factory.create_backbone(kind, config)`,
  plus `ModelConfig.backbone`, `cond_dim`, `n_channels`, `base`, `t_dim` (all optional,
  default-valued — no effect on `GanConfig` or other model configs that don't use this
  pattern). Lets `Model.__init__` resolve a pluggable backbone (`nn.Module`) from
  `config.model.backbone` instead of every model subclass hardcoding its own architecture.
  `Model.forward(*args, **kwargs)` delegates to the resolved backbone when one is set;
  subclasses with their own architecture (e.g. `WGANModel`) simply override `forward` as
  before and are unaffected.
- `DDPMConfig.beta_start` / `beta_end` — the noise schedule bounds passed to
  `torch.linspace(beta_start, beta_end, n_times)` are now configurable instead of hardcoded
  to `(1e-4, 0.02)`. Matters once `n_times` is reduced: those defaults are calibrated for
  `n_times=1000` (`alphas_cumprod[n_times-1] ≈ 4e-5`, i.e. near-total noise at the last
  training step, matching the pure noise that sampling actually starts from). With a smaller
  `n_times` and the same bounds, `alphas_cumprod[n_times-1]` stays far from zero (e.g. `≈0.36`
  at `n_times=100`) — training never sees an input as noisy as the real starting point of
  generation, a train/inference mismatch that degrades sample quality. Rule of thumb:
  `beta_end ≈ 0.02 * 1000 / n_times` keeps `alphas_cumprod[n_times-1]` at roughly the same
  order of magnitude as the `n_times=1000` default.
- `DDPM_LOSS.VALIDATE` and a generic `DDPM._inner_validate` — `DDPM` previously had no
  validation loop at all (unlike other `Training` subclasses), so a validation loader passed
  to `fit()` was silently unused for it.
- `Logger.report_metric_history(key, points)` — logs a full `(step, value)` series for one
  metric key via one (or a few, chunked to MLflow's `MAX_METRICS_PER_BATCH`)
  `MlflowClient.log_batch` call, instead of one `mlflow.log_metrics` call per point.
- `DDPM._inner_train` / `_inner_validate` now feed `self._metrics["Train"]` /
  `self._metrics["Valid"]` with `x0_hat` (the one-step reconstruction already computed for
  the loss), mirroring what `FlowMatchingTraining` already does with `x1_hat`. Metrics
  configured via `config.metrics` are now tracked for `DDPM` runs too, not just Flow
  Matching — previously silently skipped since `DDPM._inner_train`/`_inner_validate` never
  called `Metrics.update(...)` at all.
- `TransformPhase` (`iapytoo.dataset.transform`, `IntEnum`: `INPUT` / `OUTPUT` / `BOTH`) and
  `Transform.phase` (class attribute, defaults to `INPUT` — retrocompatible, every existing
  `Transform` subclass keeps its current behavior unless it explicitly overrides `phase`).
  `MlflowModel.predict` (`iapytoo.mlflow.model`) now consults it: forward-transforms the
  input (`self.transform(batch)`) only when `phase` is `INPUT`/`BOTH` (the historical,
  unconditional behavior — correct for a classic predictive model whose input is real
  physical data needing normalization before the forward pass), and applies the inverse to
  the generated output (`self.transform.inv(predictions)`) when `phase` is `OUTPUT`/`BOTH`.
  Needed for generative models (`DDPMModel`/`FlowMatchingModel.evaluate_one`): their input is
  already noise in the model's native space (forward-transforming it would corrupt the
  sampling process — verified empirically in the sibling project, noise `std` dropping from
  `~1.0` to `~0.47` after an unconditional forward-transform), and only the generated output
  needs denormalizing back to physical units.
- `MlInput.condition: Optional[MlConditionInput]` — lets each `predict()` example carry its own
  conditioning vector (`c`) alongside its main array (`x`), enabling conditional generation
  (e.g. `FlowMatchingModel.evaluate_one(x, c=...)`) through the MLflow pyfunc artifact alone,
  with no need to reload a raw checkpoint. `MlInput.from_array(array, condition=...)` builds it
  (embeds a nested `MlConditionInput.from_array(condition)`); `to_condition_array(context)` is
  the `condition`-side counterpart of `to_array(context)`. `MlflowModel.predict` builds a
  batched `c` tensor from `model_input`'s conditions (all-`None` → `c=None`, unchanged
  non-conditional path; all-set → stacked `c` tensor passed to `evaluate_one(x, c=...)`; a mix
  raises, since that's a caller mistake rather than a case to paper over) — 100% backward
  compatible for models whose `MlInput`s never set `condition`.
  Deliberately *not* implemented via `predict()`'s existing `params: dict` argument: `params`
  is one dict per HTTP request/call, applied to the whole batch, with no alignment mechanism
  to `model_input`'s rows (`ParamSchema`/`ParamSpec` support scalars or flat lists, not
  per-row structure) — fine for a hyperparameter constant across a batch, structurally wrong
  for a condition that could vary per sample. `MlInput` is already the "one entry per sample"
  structure, so conditioning belongs there.
  HTTP-serving note: `list[MlInput]` today only round-trips over HTTP via MLServer's v2/KServe
  protocol + `iapytoo/mlflow/codec.py`'s hand-written codec (`MlInputCodec`/`MlRequestCodec`,
  updated here to also encode/decode the nested `condition`) — MLflow's native scoring server
  (`mlflow models serve`, `dataframe_split`/`instances` JSON) has never round-tripped a raw
  `MlInput` (verified: it materializes plain DataFrames/records, not custom pydantic types),
  conditional or not, so this adds no new HTTP-serving gap.
  `condition` is typed `Optional[MlConditionInput]`, a **new base class** that `MlInput` now
  extends (`MlInput(MlConditionInput)`, one-way inheritance) rather than `Optional["MlInput"]`
  (self-reference) as first tried: a self-referencing type broke `save_mlflow_model()` outright
  — MLflow's type-hint-based schema walker
  (`mlflow.models.signature._infer_signature_from_type_hints`, unconditionally invoked from
  `mlflow.pyfunc.log_model`) recurses into nested pydantic models with no cycle detection, hits
  Python's real recursion limit (`RecursionError`), and that particular MLflow call site doesn't
  catch generic exceptions the way `_get_func_info_if_type_hint_supported` does elsewhere in the
  same library (it assumes `e.message`, which `RecursionError` doesn't have, raising a secondary
  `AttributeError` that aborts `log_model()` entirely) — verified by actually running a training
  script through `save_mlflow_model()`, not just importing the module.
  `MlConditionInput` holds the `on_disk`/`data` fields, the `ensure_bytes` validator, the `path`
  property, and the base (de)serialization logic (`from_array`/`to_array`/`to_bytes`) shared by
  both classes; `MlInput` inherits all of that and adds only what's genuinely `MlInput`-specific
  — the `condition` field itself, the on-disk `input_example()` artifact-placeholder resolution
  in `to_array` (a condition is always embedded via `from_array`, never registered as that
  placeholder, so `MlConditionInput.to_array`'s plain on-disk branch never needs it), and
  `to_condition_array`. This avoids the cycle at the type level (walking `MlInput.condition`
  only ever reaches `MlConditionInput`'s scalar fields, never `MlInput` itself — the inheritance
  is strictly one-directional) while removing the duplication an earlier, non-inheriting version
  of this same split had (two independent classes both declaring `on_disk`/`data`/the validator/
  `path`/`to_array`'s common branch). `MlConditionInput` must stay the hierarchy's terminal node:
  never give it a field that references `MlInput` or `MlConditionInput`, or the cycle comes back.
  `cond_indices: Optional[list[int]]` / `cond_labels: Optional[list[str]]` added to
  `ModelConfig` (same optional/no-default-impact pattern as `cond_dim`/`backbone`) so a
  conditional model's expected conditioning columns/order travel with `config.yaml` in the
  MLflow artifact instead of staying flat in a training YAML only the training script reads.
  DDPM conditional models aren't wired into this path yet: `DDPMModel.evaluate_one`
  (`iapytoo/train/model.py`) doesn't accept a `c` argument at all (unlike
  `FlowMatchingModel.evaluate_one`), so today only Flow Matching conditional benefits — left
  as-is rather than extended speculatively, since no active training script in the sibling
  project produces a DDPM-conditional artifact to test against yet.
- `iapytoo/mlflow/mlserver_runtime.py` (`MLflowRuntime`) — a thin subclass of
  `mlserver_mlflow.MLflowRuntime` whose only job is to import `iapytoo.mlflow.codec` as a side
  effect before delegating everything else. Needed because a real `mlserver` deployment
  (`model_settings.json` → `"implementation": "mlserver_mlflow.MLflowRuntime"`) never decodes a
  `content_type="mlmodelinput"` request into `list[MlInput]` on its own: `MLflowRuntime.predict`
  calls `self.decode_request(payload)` with no `default_codec`, and codec resolution there is a
  pure registry lookup by content type, populated only when `@register_input_codec`/
  `@register_request_codec` (`MlInputCodec`/`MlRequestCodec`) actually run — which requires
  importing `iapytoo.mlflow.codec` somewhere in that process. Nothing does: `iapytoo/mlflow/
  __init__.py` is empty, `MlflowModel.from_context` only imports the *provider*'s module (never
  `codec`), `mlserver_mlflow` itself has no idea this iapytoo-specific codec exists, and iapytoo
  declares no `entry_points` for mlserver's plugin discovery. Without the import, `decode_request`
  silently falls back to the raw `InferenceRequest` object, which MLflow's own type-hint
  validation then rejects (`Expected list, but got InferenceRequest` — `predict`'s signature
  declares `model_input: list[MlInput]`) — reproduced against a real deployed model, unrelated to
  conditioning (`MlInput.condition`): it would already fail for a plain, non-conditional
  `MlInput` the moment `content_type="mlmodelinput"` is actually used end-to-end through a real
  `mlserver` process rather than only exercised client-side (`test_infer_codec.py` importing the
  codec only registers *encoding*, in the client process — it does nothing for the server's
  decoding). Fix: point `model_settings.json`'s `"implementation"` at
  `"iapytoo.mlflow.mlserver_runtime.MLflowRuntime"` instead.

### Changed

- `Training._report_metrics` now flushes each loss type's buffered points through
  `Logger.report_metric_history` in a single call, instead of looping and calling
  `Logger.report_metric` once per point. With `plotting_mean` in `("mean", "ewm")` — required
  whenever `checkpoint_epoch` is set, cf. `Config.validate_config` — that buffer holds every
  raw per-batch point accumulated since the last flush (only at checkpoints, so potentially
  thousands of points). Logging them one MLflow call at a time made each checkpoint save
  noticeably slower, and increasingly so as the run's on-disk metric history grew (observed:
  a checkpoint's metric-logging burst going from ~31s to ~160s over 9 checkpoints on the same
  point count, on a local `FileStore` backend).

### Fixed

- `DDPMModel.evaluate_one`: `z = torch.randn_like(x)` and
  `t_batch = torch.full((x.shape[0],), t)` were created without `device=x.device`, always
  landing on CPU regardless of where the model/input actually live — a latent CPU/GPU
  mismatch during sampling.
- `DDPMModel.evaluate_one` and `MlflowModel.predict` neither called `self.eval()` nor wrapped
  their loop/call in `torch.no_grad()`, unlike the base `Model.evaluate_one` which does both.
  Values generated were unaffected here (the reference backbone only uses
  `GroupNorm`/`SiLU`, no `BatchNorm`/`Dropout`, so train vs eval mode doesn't change outputs —
  verified by re-running generation with a controlled RNG seed and diffing bit-for-bit), but
  without `torch.no_grad()` the output carried a live autograd graph
  (`requires_grad=True`, `grad_fn=AddBackward0`) spanning all `T` sequential reverse steps,
  never freed since `.backward()` is never called — measured ~1.16-1.19x slower on a small
  backbone (`base=16`, 116k params, CPU, batch=4) and expected to scale worse (memory
  especially) on larger backbones/batches or GPU. `DDPMModel.evaluate_one` now calls
  `self.eval()` and wraps its loop in `torch.no_grad()`, matching the base class contract;
  `MlflowModel.predict` (the pyfunc serving path, used by e.g. an mlserver deployment) also
  wraps its `evaluate_one` call in `torch.no_grad()` as defense in depth for any model type
  whose own `evaluate_one` might not self-guard.
- `save_mlflow_model` passed `input_example=[MlInput.input_example()]` (the on-disk placeholder,
  `to_array` resolving via `context.artifacts`) to `mp.log_model(...)`, which triggers MLflow's
  own automatic signature inference — that calls `predict(context=None, input_example)` with a
  bare `context=None` (no `PythonModelContext` exists yet at that point in the save flow). With
  the on-disk placeholder, `to_array(None)` hits `context.artifacts` on a `None` context
  (`AttributeError`), which `mlflow.models.signature` catches and downgrades to a warning
  ("Failed to run the predict function on input example") — silent otherwise, no traceback
  unless logging is at DEBUG (diagnosed via `mlflow.models.validate_serving_input`, which
  succeeds because it goes through a real reloaded context, unlike the in-process signature
  check).
  The on-disk placeholder is deliberate for *large* input examples (e.g. a 100x100x100 cube):
  embedding it directly (`MlInput.from_array`, `on_disk=False`, bytes inlined in the object)
  would avoid the `context=None` problem entirely (`to_array` needs no context at all in that
  branch) but bloats the logged model's metadata for a large array. Fixed with a size split
  instead of unconditionally switching mechanisms:
  `MlflowModelProvider.INPUT_EXAMPLE_EMBED_MAX_BYTES` (1 MiB — generous for a typical
  time-series/tabular example, e.g. ~1.5 KB for a `(3, 128)` float32 signal, well under any
  cube-sized example) — `provider.input_example.nbytes` at or below that threshold uses
  `MlInput.from_array` (embedded, so signature inference now succeeds instead of warning);
  above it, keeps the on-disk placeholder as before (signature inference still warns for that
  case — accepted tradeoff, avoids embedding a large array — but the artifact still loads and
  resolves normally through a real context at model load time, so the model itself is
  unaffected). The on-disk artifact registration (`artifacts["input_example"]`) is unchanged in
  both cases, so `MlInput.input_example()` still resolves correctly against a real context
  regardless of which branch was used for `input_example=`.

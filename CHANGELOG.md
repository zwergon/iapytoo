# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

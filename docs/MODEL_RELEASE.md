# Model release checklist

Current recommended release: `v12-geom-base`

## Asset contents

The minimum asset for current inference is the v12 checkpoint:

```text
models_medium_improv/highorder_signed_v12_geom_mom1e4.pt
```

The matching config should be committed to the repository:

```text
configs/v20b_base_a3p0_mink8_geom_v12.json
```

Suggested asset name:

```text
remapgnn_v12_geom_base_2026-06-29.tar.gz
```

Create the asset from the repository root:

```bash
tar -czf remapgnn_v12_geom_base_2026-06-29.tar.gz \
  models_medium_improv/highorder_signed_v12_geom_mom1e4.pt
```

Then upload it to a GitHub release at:

```text
https://github.com/MiloHS/remapgnn/releases
```

Suggested tag:

```text
v12-geom-base
```

## Release notes text

```text
Current RemapGNN research-prototype checkpoint.

Default model: v12_geom_base
Config: configs/v20b_base_a3p0_mink8_geom_v12.json
Inference projection: float64, eps_rel=1e-12, n_cg=800

This model is conservative to about 2e-9 under the cleaned projection on the
tested non-ICO audit set. It is not more accurate than TempestRemap np2, and it
does not beat cached Tempest maps. Its useful comparison point is faster
new-operator construction than the tested TempestRemap np2 generation path.
```

## After publishing

Update `docs/INFERENCE.md` with the final release URL and download command.

The old v18 release should remain available as a legacy artifact, but it should
not be described as the current default model.

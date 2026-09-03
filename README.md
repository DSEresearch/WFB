## Wave Function Backpropagation with Explicit Temporal-Interval Dynamics

Conventional neural networks learn predominantly through affine transformations followed by nonlinear activations, while elapsed time is often treated as an auxiliary feature or assumed to be uniformly sampled. This paper introduces Wave Function Backpropagation (WFB), a wave-parameterized learning formulation in which neural responses are represented by learnable amplitude, wavenumber, angular frequency, and phase. The formulation associates an observed state with its temporal interval Delta t through the phase of a differentiable spatiotemporal wave. We derive standard WFB gradients and a spatial-curvature correction based on the Laplacian of the wave response. WFB is instantiated in a deliberately feed-forward trajectory predictor to provide a controlled proof of concept; sequence learning is outside the scope of the present evaluation. With motion features, STD-WFB using real intervals reduces average displacement error (ADE) by 20.4% relative to the original FFN baseline. In a new position-only evaluation that removes temporal leakage through precomputed velocity and acceleration, real-interval WFB reduces ADE by 10.4% relative to the original FFN and remains competitive with parameter-matched ReLU controls, obtaining 2.1% lower mean ADE than the matched FFN with explicit Delta t. Shuffled-interval WFB attains the lowest mean ADE, indicating that the present evidence supports the effectiveness of the wave representation but does not attribute the gain to interval alignment. These results establish WFB as a viable structured feed-forward learning formulation and define a clear basis for subsequent architectural studies.


This project compares a standard feed-forward network (FFN) against WFB-FFN variants for pedestrian/agent trajectory prediction on ETH/UCY and JAAD-style annotations.

The central test is whether explicit temporal intervals help:

- `ffn`: flattened motion-state sequence only
- `ffn_dt`: the same ReLU FFN with standardized `delta_t` concatenated explicitly
- `ffn_matched` / `ffn_dt_matched`: ReLU controls width-matched to the WFB parameter count
- `sine_ffn_dt_matched`: parameter-matched SIREN-style periodic-activation control
- `wfb_real_t`: WFB-FFN with measured `delta_t`
- `wfb_shuffled_t`: WFB-FFN with one deterministic per-window interval permutation
- `wfb_constant_t`: WFB-FFN with the training-set mean interval

## Recommended Proof-of-Concept Evaluation

The review-oriented runner stays entirely feed-forward: it does not train a
recurrent, attention, Transformer, or other sequence-learning model. It runs
position-only temporal controls, motion controls with velocity/acceleration
recomputed after interval interventions, validation-selected Laplacian sweeps,
and the five-seed decoder ablation:

```bash
python scripts/run_proof_of_concept_suite.py \
  --processed_dir outputs/preprocessed_irregular \
  --output_dir outputs/proof_of_concept_review \
  --seeds 1 2 3 4 5 \
  --device cuda
```

Use `--suites position_controls` for the smallest clean temporal experiment.
The position-only suite prevents real-interval information from entering through
precomputed velocity or acceleration. In the motion suite,
`--motion_dt_policy recompute` recomputes those features using the intervened
intervals instead of silently retaining their real-interval values.

For a stricter source/scene-disjoint split, preprocess with `--split_unit source`.
This requires at least three distinct sources. The original paper protocol is
available as `--split_unit source_agent`.

To compare the paper-style WFB variants and sweep Laplacian lambda values:

```bash
python scripts/run_experiment.py \
  --processed_dir outputs/preprocessed_irregular \
  --output_dir outputs/wfb_variant_sweep \
  --models wfb_real_t \
  --wfb_variants standard laplacian combined \
  --lambda_laplacians 1e-7 1e-6 1e-5 1e-4 1e-3 \
  --seeds 1 2 3 4 5 \
  --epochs 100 \
  --batch_size 512 \
  --device cuda

python scripts/summarize_lambda_sweep.py \
  --results_csv outputs/wfb_variant_sweep/results_by_seed.csv
```
<!-- > python .\WFB_GoEmostions.py --output_dir outputs -->
<!--
<img width="584" height="459" alt="laplacian" src="https://github.com/user-attachments/assets/b6303b1b-6f15-4683-a9fe-8c9a18b8315e" />
-->
Full Paper: https://arxiv.org/abs/2609.00503 

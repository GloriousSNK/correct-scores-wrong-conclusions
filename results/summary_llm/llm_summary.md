# Combined LLM physics-forecasting: statistical results

- Part A cells: **2160** (success 92.2%)  |  Part B cells: **1800** (success 94.9%)
- Reliability-adjusted error (failed cell = pi/2 rad). Paired Wilcoxon / unpaired Mann-Whitney; 95% CIs by bootstrap (B=10000); p_holm = Holm within each family.
- **Total estimated API cost: $110.38**

## Part A - cross-model ranking (paired)
- `deepseek-v4-pro_minus_grok-4-1-fast-reasoning` [overall, paired]: mean diff +0.0697 rad (95% CI [+0.0176, +0.1214]), r=+0.096, n=720, p_holm=0.1920 -> n.s.
- `deepseek-v4-pro_minus_kimi-k2.6` [overall, paired]: mean diff +0.3332 rad (95% CI [+0.2759, +0.3916]), r=+0.481, n=720, p_holm=0.0000 -> **significant** (lower error for second term)
- `grok-4-1-fast-reasoning_minus_kimi-k2.6` [overall, paired]: mean diff +0.2635 rad (95% CI [+0.2146, +0.3131]), r=+0.537, n=720, p_holm=0.0000 -> **significant** (lower error for second term)

## Part A - chain-of-thought (cot - no_cot, paired)
- `cot_minus_nocot` [deepseek-v4-pro, paired]: mean diff -0.0415 rad (95% CI [-0.1064, +0.0215]), r=-0.085, n=360, p_holm=0.3418 -> n.s.
- `cot_minus_nocot` [grok-4-1-fast-reasoning, paired]: mean diff -0.0380 rad (95% CI [-0.0999, +0.0261]), r=-0.142, n=360, p_holm=0.2078 -> n.s.
- `cot_minus_nocot` [kimi-k2.6, paired]: mean diff +0.1210 rad (95% CI [+0.0605, +0.1830]), r=+0.132, n=360, p_holm=0.2078 -> n.s.
- `cot_minus_nocot` [overall, paired]: mean diff +0.0138 rad (95% CI [-0.0228, +0.0504]), r=-0.029, n=1080, p_holm=0.4504 -> n.s.

## Part A - regime contrasts (UNPAIRED; confounded - see Part B)
Negative diff = changed_hidden has LOWER error. ICs differ across regimes, so disclosure and constants are conflated here.
- `hidden_minus_disclosed` [k1, unpaired]: mean diff -0.5182 rad (95% CI [-0.6254, -0.4151]), r=-0.201, n=480, p_holm=0.0011 -> **significant** (lower error for first term)
- `hidden_minus_disclosed` [k2, unpaired]: mean diff -0.2379 rad (95% CI [-0.3341, -0.1379]), r=-0.136, n=480, p_holm=0.0688 -> n.s.
- `hidden_minus_disclosed` [k3, unpaired]: mean diff -0.4147 rad (95% CI [-0.5091, -0.3184]), r=-0.449, n=480, p_holm=0.0000 -> **significant** (lower error for first term)
- `hidden_minus_disclosed` [overall, unpaired]: mean diff -0.3903 rad (95% CI [-0.4528, -0.3275]), r=-0.274, n=1440, p_holm=0.0000 -> **significant** (lower error for first term)
- `hidden_minus_normal` [k1, unpaired]: mean diff -0.4473 rad (95% CI [-0.5619, -0.3389]), r=+0.108, n=480, p_holm=0.2078 -> n.s.
- `hidden_minus_normal` [k2, unpaired]: mean diff -0.6423 rad (95% CI [-0.7461, -0.5363]), r=-0.503, n=480, p_holm=0.0000 -> **significant** (lower error for first term)
- `hidden_minus_normal` [k3, unpaired]: mean diff -0.4814 rad (95% CI [-0.5769, -0.3831]), r=-0.494, n=480, p_holm=0.0000 -> **significant** (lower error for first term)
- `hidden_minus_normal` [overall, unpaired]: mean diff -0.5237 rad (95% CI [-0.5908, -0.4543]), r=-0.325, n=1440, p_holm=0.0000 -> **significant** (lower error for first term)

## Part B - matched hidden-disclosed (clean paired contrast)
SAME trajectories, only disclosure toggled. Negative diff = hidden has LOWER error (in-context system-ID benefit).
- `hidden_minus_disclosed` [overall, paired]: mean diff +0.0688 rad (95% CI [+0.0225, +0.1149]), r=+0.101, n=900, p_holm=0.0344 -> **significant** (lower error for second term)
- `hidden_minus_disclosed` [k1, paired]: mean diff +0.2969 rad (95% CI [+0.2077, +0.3866]), r=+0.512, n=300, p_holm=0.0000 -> **significant** (lower error for second term)
- `hidden_minus_disclosed` [k2, paired]: mean diff -0.0353 rad (95% CI [-0.1189, +0.0472]), r=-0.065, n=300, p_holm=0.4676 -> n.s.
- `hidden_minus_disclosed` [k3, paired]: mean diff -0.0552 rad (95% CI [-0.1092, -0.0018]), r=-0.204, n=300, p_holm=0.0164 -> **significant** (lower error for first term)
- `hidden_minus_disclosed` [h1, paired]: mean diff +0.1446 rad (95% CI [+0.0871, +0.2023]), r=+0.282, n=450, p_holm=0.0000 -> **significant** (lower error for second term)
- `hidden_minus_disclosed` [h10, paired]: mean diff -0.0071 rad (95% CI [-0.0765, +0.0641]), r=-0.070, n=450, p_holm=0.4676 -> n.s.

## Part B - inference probe (inferred vs true constants)
Per model x constant: MAE and Spearman(inferred, true) with 95% CI. L/m pooled across link positions. A `nan` Spearman means the model emitted a CONSTANT value for that constant (no variation -> correlation undefined): i.e. it fell back on an Earth-standard prior (g~9.81, L=m=1.0) instead of inferring. ACCURACY_VS_INFERENCE = Spearman(constant-inference error, forecast error): positive => better system-ID tracks lower forecast error.
| model | constant | n | mae | spearman | spearman_ci_lo | spearman_ci_hi | spearman_p | pearson |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-v4-pro | g | 300 | 1.8064 | nan | nan | nan | nan | nan |
| deepseek-v4-pro | damping | 300 | 0.0921 | -0.0268 | -0.1445 | 0.0882 | 0.6436 | -0.0351 |
| deepseek-v4-pro | L | 600 | 0.2494 | 0.1096 | 0.0531 | 0.1652 | 0.0072 | 0.1149 |
| deepseek-v4-pro | m | 600 | 0.4950 | -0.0200 | -0.0991 | 0.0606 | 0.6256 | 0.0009 |
| grok-4-1-fast-reasoning | g | 288 | 1.8754 | 0.0169 | -0.0871 | 0.1127 | 0.7753 | -0.0409 |
| grok-4-1-fast-reasoning | damping | 288 | 1.3027 | -0.0498 | -0.1651 | 0.0720 | 0.3995 | 0.0140 |
| grok-4-1-fast-reasoning | L | 580 | 0.2523 | nan | nan | nan | nan | nan |
| grok-4-1-fast-reasoning | m | 580 | 0.4864 | nan | nan | nan | nan | nan |
| kimi-k2.6 | g | 300 | 2.1166 | 0.0216 | -0.1075 | 0.1419 | 0.7098 | 0.0643 |
| kimi-k2.6 | damping | 300 | 0.1252 | -0.0298 | -0.1489 | 0.0837 | 0.6075 | -0.0564 |
| kimi-k2.6 | L | 600 | 0.2513 | nan | nan | nan | nan | nan |
| kimi-k2.6 | m | 600 | 0.4877 | nan | nan | nan | nan | nan |
| deepseek-v4-pro | ACCURACY_VS_INFERENCE | 300 | nan | -0.0970 | -0.2076 | 0.0164 | 0.0936 | nan |
| grok-4-1-fast-reasoning | ACCURACY_VS_INFERENCE | 288 | nan | 0.2022 | 0.0916 | 0.3074 | 0.0006 | nan |
| kimi-k2.6 | ACCURACY_VS_INFERENCE | 300 | nan | 0.0311 | -0.0790 | 0.1404 | 0.5919 | nan |

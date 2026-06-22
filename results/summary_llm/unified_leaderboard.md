# Unified cross-family leaderboard (out-of-sample)

- Canonical cell grid: **360** cells (movement_id x horizon in {1,10}s), all **15 models** scored on the SAME grid.
- Reliability-adjusted mean angle error (rad); failed OR missing cell = pi/2. 95% bootstrap CI (B=10000). LLMs scored on no_cot forecasts. `n_missing` = cells the model never produced.

| rank | model | family | angle_error_mean | ci_lo | ci_hi | success_rate | n | n_missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | rk4 | numerical | 0.0591 | 0.0322 | 0.0892 | 1.0000 | 360 | 0 |
| 2 | symplectic | numerical | 0.2052 | 0.1630 | 0.2505 | 1.0000 | 360 | 0 |
| 3 | euler | numerical | 0.4049 | 0.3435 | 0.4697 | 1.0000 | 360 | 0 |
| 4 | kimi-k2.6 | llm | 0.5380 | 0.4754 | 0.6022 | 0.9917 | 360 | 0 |
| 5 | chronos2-multi | time-series | 0.6843 | 0.6170 | 0.7524 | 1.0000 | 360 | 0 |
| 6 | chronos2-uni | time-series | 0.6971 | 0.6307 | 0.7659 | 1.0000 | 360 | 0 |
| 7 | neural-ode-rollout-mixed | learned | 0.8435 | 0.7702 | 0.9197 | 1.0000 | 360 | 0 |
| 8 | neural-ode-rollout | learned | 0.8701 | 0.7954 | 0.9468 | 1.0000 | 360 | 0 |
| 9 | grok-4-1-fast-reasoning | llm | 0.8810 | 0.8054 | 0.9568 | 0.8194 | 360 | 0 |
| 10 | hnn-rollout | learned | 0.9218 | 0.8472 | 0.9982 | 1.0000 | 360 | 0 |
| 11 | deepseek-v4-pro | llm | 0.9524 | 0.8806 | 1.0272 | 1.0000 | 360 | 0 |
| 12 | hnn-rollout-mixed | learned | 1.0028 | 0.9229 | 1.0844 | 1.0000 | 360 | 0 |
| 13 | neural-ode | learned | 1.0284 | 0.9526 | 1.1071 | 1.0000 | 360 | 0 |
| 14 | hnn | learned | 1.0635 | 0.9820 | 1.1450 | 1.0000 | 360 | 0 |
| 15 | lnn | learned | 1.4215 | 1.3627 | 1.4811 | 0.4528 | 360 | 197 |

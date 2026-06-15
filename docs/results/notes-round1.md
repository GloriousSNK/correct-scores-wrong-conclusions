 Findings

  1. Physics is still physics

  Numerical integrators dominate. RK4 achieves sub-milliradian error at 0.01s and
  1s horizons — essentially perfect. The gap to everything else is enormous. Even
  at 60s (where chaotic divergence should destroy any fixed-step integrator), RK4
  still beats every LLM and learned model. The takeaway: if you know the equations
  of motion, solve them. No LLM is close.

  2. Neural ODE works — and beats all LLMs cleanly

  Neural ODE (0.588 rad, 100% success) is the best non-numerical method. It learned
  a usable dynamics model from 67K trajectory samples in ~3 minutes on MPS.
  Critically, it succeeds on every single cell — no format failures, no refusals,
  no hallucinations. Compared to grok's 75% success and kimi's 7%, that reliability
  matters as much as accuracy.

  3. The LLMs are doing something, but it's fragile

  - grok at 0.01s horizon achieves near-zero angle error (~10⁻⁷ rad) on simple k=1
  normal regime cells. It's not just guessing — it understands that a 0.01s
  prediction is barely different from the initial condition. But at 1s+ horizons on
  k=3 systems, it falls apart (0.81–1.14 rad). It's interpolating near the initial
  state, not simulating dynamics.
  - kimi has a single cell with 6×10⁻¹⁰ rad error — essentially machine epsilon.
  When it responds correctly, it may be reasoning at a remarkable level. But 92.6%
  of the time it refuses or produces unparseable output. It's brilliant and
  unusable.
  - deepseek works fine on coords (98.6% success) but completely fails on images
  (0%). The model's output format for numerical predictions is inconsistent with
  the parser — a harness problem as much as a model problem.

  4. CoT doesn't help LLMs predict physics

  Across all three LLMs, CoT either makes no difference or slightly hurts accuracy.
  The reasoning chain doesn't scaffold correct simulation — LLMs can't iteratively
  integrate differential equations in their forward pass the way a numerical
  solver does. Thinking longer about physics doesn't make the physics right.

  5. HNN < Neural ODE, which is surprising

  The Hamiltonian Network has a stronger structural prior (it's forced to conserve
  energy via the H structure), but performs worse than the unconstrained Neural
  ODE. Likely cause: 500 epochs wasn't enough for the HNN loss (which requires
  autograd through the Hamiltonian) to converge as well as the simpler MSE
  derivative-matching in Neural ODE. The inductive bias isn't free — it demands
  more careful training.

  6. LNN is currently broken for this use case

  The LNN has two unsolved problems: (a) training requires second-order autograd
  which doesn't run on MPS, forcing CPU and making full-scale training take hours,
  and (b) inference via scipy solve_ivp with autograd RHS takes minutes per
  60s-horizon cell. The 8K-subsample, 64-hidden workaround gets it running but
  produces the worst accuracy of all models. The architecture is theoretically the
  most physically correct — but operationally unusable right now.

  ---
  What I Think

  The most interesting finding is the reliability gap between LLMs and Neural ODE.
  Neural ODE beats grok on both accuracy and reliability — every single prediction
  succeeds and comes out better on average. This is strong evidence that if you
  have training data, even a simple learned dynamics model is preferable to a
  frontier LLM for physics prediction.

  The kimi result is worth taking seriously. Sub-picometre angle error on a
  successful prediction suggests it may be doing actual physics reasoning in some
  cases — not lookup or interpolation. The 7.4% success rate is a prompt
  engineering problem, not a capability problem. That's fixable.

  The horizon × success interaction is the most useful scientific signal: LLMs
  perform better at 60s than 1s on angle error in some cases. That's almost
  certainly because at 60s, chaotic sensitivity means any answer is roughly as bad
  as any other — the error floor is high for everyone — while at 1s the system is
  still sensitive enough that wrong physics produces high error. LLMs aren't
  getting better at long horizons; the problem is getting easier to accidentally
  get right.

  ---
  Next Steps

  High priority:

  1. Fix kimi's output format — enforce JSON output via system prompt or Azure
  inference parameters. If kimi at 100% success rate is anywhere near 0.287 rad, it
  becomes the best LLM result by a wide margin.
  2. Fix deepseek's image modality — it currently has 0% success on
  images/images_coords. Almost certainly a prompt or system message issue.
  DeepSeek-V4-Pro does support vision; fix the parser or format string.
  3. Add gpt-5.5 and qwen3-vl-32b (Round 2, needs AWS) — the vision-capable
  reasoning models are the most interesting comparison point, and both were
  deferred only due to infrastructure, not model capability.

  Medium priority:

  4. Retrain LNN properly — either find a CPU server with enough cores, or rewrite
  the E-L loss to avoid create_graph=True by pre-computing mass matrix targets
  numerically. The architecture is the most physically principled; it deserves a
  real training run.
  5. Add time-series models (Chronos, TimesFM, Moirai) — they're in the config but
  not deployed. These treat the trajectory as a sequence and make no physics
  assumptions, which is a different and interesting baseline.
  6. Train on more data / longer — Neural ODE and HNN were both trained for 500
  epochs on 67K samples. Scaling to 200+ epochs with a learning rate schedule tuned
  for each model would likely close the HNN–Neural ODE gap and push Neural ODE
  error down.

  Longer term:

  7. Run full trajectory prediction, not just endpoint prediction — currently we
  predict the state at time T from the initial state. Predicting the full
  trajectory and computing divergence time would be a richer benchmark, especially
  for chaos analysis.
  8. Add a best of N or self-consistency metric for LLMs — sample k=5 responses,
  take the median prediction. This might dramatically improve effective LLM
  accuracy and is a realistic deployment strategy.

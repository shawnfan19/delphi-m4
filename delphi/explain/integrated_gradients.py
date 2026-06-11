import torch


def integrated_jacobian(
    func,
    inputs: torch.Tensor,
    baselines: torch.Tensor,
    n_steps: int = 50,
    mode: str = "forward",
    chunk_size: None | int = None,
) -> torch.Tensor:
    """Compute integrated gradients of a vector-valued function.

    A single-scalar input (``inputs.numel() == 1``) short-circuits to the exact
    endpoint difference ``func(inputs) - func(baselines)``: the path is 1-D, so by
    the fundamental theorem of calculus the integral is redundant (and would only
    add quadrature error). Two forward passes, no Jacobian.

    Parameters
    ----------
    func : callable
        Differentiable function that takes a tensor of the same shape as
        *inputs* and returns a 1-D tensor of shape ``(n_outputs,)``.
    inputs : torch.Tensor
        The point at which to explain.
    baselines : torch.Tensor
        Reference point (same shape as *inputs*).
    n_steps : int
        Number of interpolation steps for the Riemann (trapezoidal) approximation.
    mode : str
        ``"forward"`` uses ``jacfwd`` (cost scales with n_inputs),
        ``"reverse"`` uses ``jacrev`` (cost scales with n_outputs).
    chunk_size : int or None
        How many interpolation points to evaluate per ``vmap`` call. The
        Jacobians at the ``n_steps + 1`` points are independent, so they are
        batched through ``vmap(jac_fn(func))`` instead of a Python loop — far
        better device utilization than one batch-1 forward per point. ``None``
        evaluates all points in a single ``vmap`` call (peak activation memory
        ~ ``(n_steps + 1) x`` a single point); set a finite ``chunk_size`` to
        cap the in-flight batch when n_inputs or the model is large. ``vmap``
        computes only the per-point (block-diagonal) Jacobian, never the
        wasteful cross-point terms a Jacobian over a batched input would.

    Returns
    -------
    torch.Tensor
        Integrated Jacobian, shape ``(n_outputs, *input_shape)``.
    """
    # single-scalar input: IG = exact endpoint difference (see docstring). Skip the
    # whole jacfwd/vmap path; shape to the full-IG layout (n_outputs, *input_shape).
    if inputs.numel() == 1:
        with torch.no_grad():
            ig = func(inputs) - func(baselines)
        return ig.reshape(-1, *inputs.shape)

    jac_fn = (
        torch.func.jacfwd if mode == "forward" else torch.func.jacrev
    )  # type: ignore[attr-defined]
    delta = inputs - baselines

    # interpolation points along the straight path baseline -> inputs, stacked on
    # a leading axis: (n_steps + 1, *input_shape). alphas broadcasts over the input
    # dims so each slice interps[k] has the same shape `func` expects.
    alphas = torch.linspace(0, 1, n_steps + 1, device=inputs.device)
    alphas = alphas.reshape(-1, *([1] * inputs.dim()))
    interps = (baselines + alphas * delta).detach()  # (n_steps + 1, *input_shape)

    # vmap's own chunk_size splits the mapped (step) axis into chunks, vmaps each,
    # and concatenates — capping in-flight batch (and activation memory). None = no
    # chunking. Verified numerically identical to a manual chunk loop on torch 2.3.
    batched_jac = torch.func.vmap(jac_fn(func), chunk_size=chunk_size)  # type: ignore[attr-defined]
    jacs = batched_jac(interps)  # (n_steps + 1, n_outputs, *input_shape)

    ig = delta * torch.trapezoid(jacs, dx=1.0 / n_steps, dim=0)
    return ig

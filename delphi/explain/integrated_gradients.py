import torch


def integrated_jacobian(
    func,
    inputs: torch.Tensor,
    baselines: torch.Tensor,
    n_steps: int = 50,
    mode: str = "forward",
) -> torch.Tensor:
    """Compute integrated gradients of a vector-valued function.

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
        Number of interpolation steps for the Riemann approximation.
    mode : str
        ``"forward"`` uses ``jacfwd`` (scales with n_inputs),
        ``"reverse"`` uses ``jacrev`` (scales with n_outputs).

    Returns
    -------
    torch.Tensor
        Integrated Jacobian, shape ``(n_outputs, *input_shape)``.
    """
    jac_fn = (
        torch.func.jacfwd if mode == "forward" else torch.func.jacrev
    )  # type:ignore
    delta = inputs - baselines
    alphas = torch.linspace(0, 1, n_steps + 1, device=inputs.device)

    jacs = []
    for alpha in alphas:
        interp = baselines + alpha * delta
        interp = interp.detach().requires_grad_(True)
        jac = jac_fn(func)(interp)  # (n_outputs, *input_shape)
        jacs.append(jac)

    jacs = torch.stack(jacs)  # (n_steps + 1, n_outputs, *input_shape)
    ig = delta * torch.trapezoid(jacs, dx=1.0 / n_steps, dim=0)
    return ig

import torch

class SingleExponential(torch.autograd.Function):
    """Surrogate gradients for standard binary spikes"""
    @staticmethod
    def forward(
            ctx,
            input,
            threshold=1.0,
            window=0.5,
            max_spikes_per_dt=torch.tensor(1.),
    ):
        ctx.save_for_backward(input.clone())
        ctx.threshold = threshold
        return input.ge(threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        threshold = ctx.threshold
        grad_input = grad_output.clone()
        return grad_input * torch.exp(-torch.abs(input - threshold)), grad_input * -torch.exp(-torch.abs(input - threshold)), None, None


class PeriodicExponential(torch.autograd.Function):
    """
    Subtract from membrane potential on reaching threshold
    """

    @staticmethod
    def forward(
        ctx,
        data,
        threshold=1.0,
        window=0.5,
        max_spikes_per_dt=torch.tensor(float("inf")),
    ):
        ctx.save_for_backward(data.clone())
        ctx.threshold = threshold
        ctx.window = window
        ctx.max_spikes_per_dt = max_spikes_per_dt
        nr_spikes = ((data >= threshold) * torch.floor(data / threshold)).float()
        nr_spikes[nr_spikes > max_spikes_per_dt] = max_spikes_per_dt.float()
        return nr_spikes

    @staticmethod
    def backward(ctx, grad_output):
        (membranePotential,) = ctx.saved_tensors

        vmem_shifted = membranePotential - ctx.threshold / 2
        nr_spikes_shifted = torch.clamp(torch.div(
            vmem_shifted, ctx.threshold, rounding_mode="floor"
        ), max=ctx.max_spikes_per_dt - 1)

        vmem_periodic = vmem_shifted - nr_spikes_shifted * ctx.threshold
        vmem_below = vmem_shifted * (membranePotential < ctx.threshold)
        vmem_above = vmem_periodic * (membranePotential >= ctx.threshold)
        vmem_new = vmem_above + vmem_below
        spikePdf = (
            torch.exp(-torch.abs(vmem_new - ctx.threshold / 2) / ctx.window)
            / ctx.threshold
        )

        return grad_output * spikePdf, grad_output * -spikePdf * membranePotential / ctx.threshold, None, None

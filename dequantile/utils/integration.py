# Copying from scipy
import torch


def tupleset(t, i, value):
    lst = list(t)
    lst[i] = value
    return tuple(lst)


def cumulative_trapezoid(y, dx):
    """Calculate the trapezoidal rule cumulatively"""
    nd = len(y.shape)
    slice1 = tupleset((slice(None),) * nd, -1, slice(1, None))
    slice2 = tupleset((slice(None),) * nd, -1, slice(None, -1))

    res = torch.cumsum(dx * (y[slice1] + y[slice2]) / 2.0, dim=-1)
    shape = list(res.shape)
    shape[-1] = 1
    res = torch.cat([torch.zeros(shape, dtype=res.dtype), res], dim=-1)
    return res


def get_integrals(inputs, outputs, y_pos):
    outputs = outputs.view(-1, 1)
    num_bins = y_pos.shape[-1] - 1
    bin_width = 1.0 / num_bins
    partition_func = torch.trapz(y_pos, dx=bin_width).view(-1, 1)

    # Get the bin index
    bin_pos = inputs * num_bins
    bin_idx = torch.floor(bin_pos).long()
    bin_idx[bin_idx >= num_bins] = num_bins - 1
    # Get the distance from the left most bin
    alpha = (bin_pos - bin_idx.float()) / num_bins

    # Get the trapezoidal rule cumulatively
    cum_trap = cumulative_trapezoid(y_pos, bin_width)
    # Define the integral over all of the preceeding bins for each input
    prev_contr = cum_trap.gather(-1, bin_idx)
    # Get the integral in the current bin
    left_bin = y_pos.gather(-1, bin_idx)
    bin_integral = (outputs + left_bin) / 2 * alpha
    # Get the cumulative integral to each point in outputs
    cumul_int = bin_integral + prev_contr

    return partition_func, cumul_int

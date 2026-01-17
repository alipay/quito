import torch


def masked_std(x, mask, dim=None, keepdim=False, unbiased=True):
    """
    Compute the standard deviation of x over specified dimensions, 
    considering only the elements where mask is True.

    Computes standard deviation while ignoring masked (False) elements.
    Useful for handling missing values in time series data.

    Args:
        x (torch.Tensor): Input tensor of shape [...]
        mask (torch.BoolTensor or torch.Tensor): Boolean mask broadcastable to x;
            True indicates valid/included element, False indicates masked/ignored.
        dim (int or tuple, optional): Dimension(s) to reduce over.
            Defaults to None (reduce over all dimensions).
        keepdim (bool, optional): Whether to keep reduced dimensions.
            Defaults to False.
        unbiased (bool, optional): Whether to use Bessel's correction (n -> n-1).
            Defaults to True.

    Returns:
        torch.Tensor: Standard deviation over masked input.
    """
    mask = mask.bool()

    # Expand mask to match x shape if needed
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)

    # Set invalid entries to 0
    x_masked = x.masked_fill(~mask, 0.0)

    # Compute sum and count of valid elements
    sum_valid = x_masked.sum(dim=dim, keepdim=keepdim)
    count_valid = mask.sum(dim=dim, keepdim=keepdim).clamp(min=1)  # avoid div by 0

    # Compute mean
    mean = sum_valid / count_valid

    # Subtract mean from original data (only valid positions matter)
    x_centered = (x - mean).masked_fill(~mask, 0.0)
    x_centered_squared = x_centered ** 2

    # Variance = mean of squared deviations over valid entries
    variance = x_centered_squared.sum(dim=dim, keepdim=keepdim) / (count_valid - unbiased * 1)
    
    # For unbiased std, we divide by (n - 1), but clamp at 1 to avoid division by zero
    if unbiased:
        variance = variance * (count_valid / (count_valid - 1).clamp(min=1))

    return torch.sqrt(variance)


def masked_mean(x, mask, dim=None, keepdim=False):
    """
    Compute mean over masked elements.
    
    Computes mean while ignoring masked (False) elements. Useful for handling
    missing values in time series data.
    
    Args:
        x (torch.Tensor): Tensor of shape [B, ..., L, ...]
        mask (torch.BoolTensor or torch.Tensor): Boolean or binary tensor
            broadcastable to x.shape[-len(mask.shape):]. True/1 means include,
            False/0 means ignore.
        dim (int or tuple, optional): Dimension(s) to reduce over.
            Defaults to None (reduce over all dimensions).
        keepdim (bool, optional): Whether to keep reduced dimensions.
            Defaults to False.

    Returns:
        torch.Tensor: Mean-reduced tensor over valid (masked) elements.
    """
    # Convert to boolean if needed
    mask = mask.bool()

    # Expand mask to match x's dimensions if necessary
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)

    # Replace non-valid values with 0, then divide by count of valid ones
    masked_x = x.masked_fill(~mask, 0.0)
    
    value_sum = masked_x.sum(dim=dim, keepdim=keepdim)
    value_count = mask.sum(dim=dim, keepdim=keepdim).clamp(min=1)  # clamp to avoid division by 0
    
    return value_sum / value_count
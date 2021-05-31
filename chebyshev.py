import torch

def chebyshev(x,n):
    """
    Implements the chebyshev polynomials using torch

    Parameters
    ----------
    x : torch.tensor of shape [N,1] and dtype torch.float32
        Points where the chebyshev polynomials should be evaluated
    n : torch.tensor of shape [M,] and dtype torch.int64
        Tells which polynomials should be returned

    Returns
    -------
    torch.tensor of shape [N,M] and dtype torch.float32
    M chebyshev polynomials evaluated for each point in x
    """
    assert len(x.shape) == 2, f"for x: need shape [N,1] but got {x.shape} -> reshape"
    assert n.dtype == torch.int64, f" for n: expected dtype torch.int64, got {n.dtype}"
    assert len(n.shape) == 1, f"for n: expected shape [M,] but got {n.shape}"

    if n.shape[0] > 1:
        result = torch.ones(x.shape[0],n.shape[0])
        mask0 = (n==0)
        mask1 = (n==1)
        result[...,mask1] = x
        mask = ((mask0 | mask1) == False)
        if n[mask].shape[0] > 0:
            result[...,mask] = 2*x * chebyshev(x,n[mask]-1) - chebyshev(x,n[mask]-2)
        return result
    else:
        if n == 0:
            return torch.ones([1])
        if n == 1:
            return x
        else:
            return 2*x*chebyshev(x,n-1) - chebyshev(x,n-2)
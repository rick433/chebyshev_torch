import torch
from chebyshev import *
import time
def chebyshev_single(x, n):
    if n == 0:
        return torch.ones([1])
    if n == 1:
        return x
    else:
        return 2 * x * chebyshev_single(x, n - 1) - chebyshev_single(x, n - 2)

def chebyshev_slow(x,n):
    ret = []
    for i in range(x.shape[0]):
        temp = []
        for j in range(n.shape[0]):
            temp.append(chebyshev_single(x[i],n[j]))
        ret.append(temp)
    return torch.tensor(ret)


def test(N,n):
    x = torch.linspace(-2*1e1,2*1e1,N).reshape(-1,1)
    start = time.time()
    result = chebyshev(x, n)
    end = time.time()
    fast_time = end - start
    start = time.time()
    control = chebyshev_slow(x, n)
    end = time.time()
    control_time = end - start
    diff = torch.abs(result - control)
    print(f"Our implementation took {fast_time:.2E}s, reference implementation took {control_time:.2E}s. Our implementation was {control_time/fast_time:.2E} x faster. \n")
    tol = 1e-6
    assert torch.all(diff<tol), f"{control} vs {result}."

if __name__=="__main__":
    test(10, torch.arange(10))
    test(1000, torch.arange(10))
    test(5, torch.tensor([20]))
    print("All tests passed.")
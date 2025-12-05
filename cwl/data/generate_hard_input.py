import numpy as np

def easy_block(n, diag=-1.0, off=0.15):
    """Sparse, almost chain-like block."""
    B = np.zeros((n, n))
    np.fill_diagonal(B, diag)
    for i in range(n - 1):
        B[i, i + 1] = off
        B[i + 1, i] = off
    return B

def hard_block(n, diag=-0.5, w=1.0):
    """Dense, sign-alternating frustrated block."""
    B = np.zeros((n, n))
    np.fill_diagonal(B, diag)
    for i in range(n):
        for j in range(i + 1, n):
            sign = 1 if (i + j) % 2 == 0 else -1
            B[i, j] = sign * w
            B[j, i] = sign * w
    return B

def assemble(blocks, inter_noise=0.02, seed=7):
    rng = np.random.default_rng(seed)
    sizes = [b.shape[0] for b in blocks]
    N = sum(sizes)
    Q = np.zeros((N, N))

    # place blocks on diagonal
    idx = 0
    ranges = []
    for b in blocks:
        n = b.shape[0]
        Q[idx:idx+n, idx:idx+n] = b
        ranges.append((idx, idx+n))
        idx += n

    # add weak sparse inter-block couplings
    for a in range(len(ranges)):
        for b in range(a + 1, len(ranges)):
            i0, i1 = ranges[a]
            j0, j1 = ranges[b]
            # sparse random links
            num_links = max(1, (i1 - i0) * (j1 - j0) // 200)
            for _ in range(num_links):
                i = rng.integers(i0, i1)
                j = rng.integers(j0, j1)
                val = rng.uniform(-inter_noise, inter_noise)
                Q[i, j] = val
                Q[j, i] = val

    return Q

if __name__ == "__main__":
    
    easy_sizes = [12] * 12
    hard_sizes = [16, 16, 12]

    blocks = [easy_block(n) for n in easy_sizes] + [hard_block(n) for n in hard_sizes]
    Q = assemble(blocks, inter_noise=0.02, seed=7)

    np.savetxt("./hybrid_big_188.csv", Q, delimiter=",", fmt="%.6f")
    print("Wrote data/hybrid_big_188.csv with shape", Q.shape)
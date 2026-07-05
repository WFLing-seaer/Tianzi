import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def get_k_ts(mask, k, TR, seed=0):  # sourcery skip: sum-comprehension
    N = len(mask)
    res_idx = np.empty((TR, k), dtype=np.int64)
    cnts = np.zeros(TR, dtype=np.int64)
    for tid in prange(TR):
        np.random.seed(seed + tid)
        start = (N * tid) // TR
        end = (N * (tid + 1)) // TR
        cnt = 0
        for i in range(start, end):
            if not mask[i]:
                continue
            if cnt < k:
                res_idx[tid, cnt] = i
            else:
                j = np.random.randint(0, cnt + 1)
                if j < k:
                    res_idx[tid, j] = i
            cnt += 1
        cnts[tid] = cnt
    np.random.seed(seed + 0x0D000721)
    total = 0
    for t in range(TR):
        total += cnts[t]
    n_out = min(k, total)
    final_idx = np.empty(n_out, dtype=np.int64)
    if n_out == 0:
        return final_idx
    rem_sel = n_out
    rem_pop = total
    out = 0
    for t in range(TR):
        c_t = cnts[t]
        if c_t == 0 or rem_sel == 0:
            continue
        m_t = min(k, c_t)
        if t == TR - 1 or rem_pop == c_t:
            n_t = rem_sel
        else:
            n_t = np.random.hypergeometric(c_t, rem_pop - c_t, rem_sel)
        n_t = min(n_t, m_t)
        idx_buf = np.arange(m_t)
        for j in range(n_t):
            r = j + np.random.randint(0, m_t - j)
            tmp = idx_buf[j]
            idx_buf[j] = idx_buf[r]
            idx_buf[r] = tmp
        for j in range(n_t):
            final_idx[out] = res_idx[t, idx_buf[j]]
            out += 1
        rem_sel -= n_t
        rem_pop -= c_t
    np.random.shuffle(final_idx)
    return final_idx


@njit(cache=True)
def find_first_true(arr):  # sourcery skip: use-next
    for i in range(len(arr)):
        if arr[i]:
            return i
    return -1


get_k_ts(np.array([], dtype=np.bool_), 1, 1, 0)
find_first_true(np.array([], dtype=np.bool_))

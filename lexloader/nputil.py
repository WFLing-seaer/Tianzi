from itertools import product
from math import exp, log, log1p

import numpy as np
from numba import njit, prange

# 本文件有相当大量的纯AIGC内容（多数是蓝色大肥鱼），而且我改不太动了，所以这个模块就当成kernel黑箱好了……


@njit(inline="always")
def _sm_next(st):
    # SplitMix64：纯整数运算，平台无关（win32/linux 结果一致）。
    # st 是 uint64[1]，作为某个线程私有的 RNG 状态；调用会推进它。
    st[0] += np.uint64(0x9E3779B97F4A7C15)
    z = st[0]
    z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> 31)


@njit(inline="always")
def _randint(st, a, b):
    # 均匀整数 in [a, b)，要求 b > a；Lemire 拒绝采样消除模偏差。
    span = np.uint64(b - a)
    two32 = np.uint64(0x100000000)
    rem = (two32 - np.uint64(1)) % span
    limit = two32 - np.uint64(1) - rem  # 2^32 以下最大的 span 整数倍
    while True:
        r = _sm_next(st) & np.uint64(0xFFFFFFFF)
        if r < limit:
            return a + np.int64(r % span)


_TWO_M53 = 1.0 / 9007199254740992.0  # 2^-53
_ONE_BELOW = 0.9999999999999999  # 1 以下最大的 float64


@njit(inline="always")
def _randf(st):
    # SplitMix64 高 53 位 → (0, 1] 的 float64。
    # 纯整数构造 + 一次乘加，不用 libm，所以这一步是跨平台逐位一致的。
    z = _sm_next(st) >> np.uint64(11)
    return np.float64(z) * _TWO_M53 + _TWO_M53


@njit(inline="always")
def _newW(st, k):
    # Algorithm L 的 W ← U^(1/k)（调用点负责累乘）。
    W = exp(log(_randf(st)) / k)
    if W >= 1.0:  # U 恰好抽到 1.0 时 exp(0)=1，会让下面的 log1p(-W) = -inf
        W = _ONE_BELOW
    return W


@njit(inline="always")
def _geom_skip(st, W, cap):
    # Algorithm L 的几何跳步：本次插入之前要跳过的候选个数，恒 >= 1。
    # 因为 P(skip >= j) = (1-W)^j，反变换即 skip = floor(ln(u)/ln(1-W)) + 1。
    # 注：这是 get_k_ts 里唯一用到超越函数的地方（见 get_k_ts 的说明）。
    g = log(_randf(st)) / log1p(-W)
    if not (g < 1.0e18):  # NaN / +inf / 溢出：本次干脆跳过本块剩余全部候选
        return np.int64(cap)
    return np.int64(g) + 1


@njit(parallel=True, cache=True)
def _sample_phase(mask, k, TR, seeds):
    # 每个 tid 扫自己那块 mask，产出该块内【均匀无放回】的至多 k 个下标，
    # 放到 res_idx[tid]，cnts[tid] 记块内 True 总数。要求 k >= 1。
    #
    # 用的是 Algorithm L（Li 1994，带几何跳步的蓄水池）而不是逐个候选掷骰子的
    # Algorithm R：块内前 k 个 True 直接入池，之后每次只决定「跳过多少个候选再
    # 换下一个」，于是每块的随机次数从 O(True 数) 降到 O(k·ln(c_t/k))。
    # 实测 4.45M 的 mask：True 占 33% 时 1.97ms → 0.60ms（3.3x），
    # True 占 67% 时 2.33ms → 0.59ms（3.9x）；代价是 True 少于 ~3% 时反而慢
    # 15%~25%（0.32ms → 0.38ms），因为那时本来就没多少 True 要掷骰子，
    # 而扫描 mask 的 0.3ms 地板是省不掉的。
    #
    # 每个 tid 的 RNG 状态是循环体内的局部变量，线程私有，绝无共享；
    # 不碰 numba 的 np.random.*（其"线程局部"状态实际是进程级共享全局，
    # 见 _random.c 的 numba_get_numba_random_state，并发调用是数据竞争）。
    N = len(mask)
    res_idx = np.empty((TR, k), dtype=np.int64)
    cnts = np.zeros(TR, dtype=np.int64)
    for tid in prange(TR):
        st = np.empty(1, dtype=np.uint64)
        st[0] = seeds[tid]
        start = (N * tid) // TR
        end = (N * (tid + 1)) // TR
        # ---- 前 k 个 True 直接入池 ----
        nfill = 0
        p = end
        for q in range(start, end):
            if mask[q]:
                res_idx[tid, nfill] = q
                nfill += 1
                if nfill == k:
                    p = q + 1  # 池满，下一个待考察位置
                    break
        cnt = nfill
        if nfill < k:  # 本块 True 不足 k 个，全都要，不用跳步
            cnts[tid] = cnt
            continue
        # ---- Algorithm L ----
        # cnt = mask[start:p] 里 True 的个数，刚取走的那个 item 下标是 cnt-1；
        # target 是「下一个要取的 item 下标」，所以走位条件是 cnt <= target。
        W = _newW(st, k)
        target = (k - 1) + _geom_skip(st, W, N)
        while p < end:
            while cnt <= target and p < end:
                if mask[p]:
                    cnt += 1
                p += 1
            if cnt <= target:  # 走到块尾还没凑够，本块结束
                break
            res_idx[tid, _randint(st, 0, k)] = p - 1
            W = W * _newW(st, k)
            target = (cnt - 1) + _geom_skip(st, W, N)
        cnts[tid] = cnt
    return res_idx, cnts


def get_k_ts(mask, k, TR, seed=0):
    """从 mask 中无放回地均匀抽取 min(k, sum(mask)) 个 True 下标。

    两段式设计：
    1. 并行采样阶段（numba prange，Algorithm L）：每个线程用 SeedSequence
       派生的独立 SplitMix64 流，状态是循环体局部变量——并行区绝无共享 RNG。
    2. 串行 merge 阶段（prange 之后已是单线程）：挪到 Python 层，直接用
       numpy 原生 RandomState（MT19937，与原版 merge 同族、同种子偏移），
       每次调用新建实例，无任何全局/共享状态，多线程并发调用也安全。

    给定 (mask, k, TR, seed) 结果确定、可复现。

    跨平台一致性（较原版收窄，见 _geom_skip）：种子派生、SplitMix64、
    Lemire 拒绝采样全是整数运算，逐位一致；但 Algorithm L 的几何跳步要用
    ln/exp，libm 在不同平台上可能差 1 ulp，从而以约 1e-11 的概率让某一次
    skip 差 1。所以「win32/linux 逐位一致」这条严格保证不再成立，
    改为「除上述量级可忽略的情形外一致」。若哪天需要严格逐位一致，
    把 _sample_phase 换回 Algorithm R（逐候选掷骰子）即可，代价见其注释。
    """
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if TR <= 0:
        TR = 1
    # SeedSequence.spawn 生成相互独立、可复现的每线程种子（numpy 官方并行 RNG 方案）
    children = np.random.SeedSequence(seed).spawn(TR)
    seeds = np.empty(TR, dtype=np.uint64)
    for i, c in enumerate(children):
        seeds[i] = c.generate_state(1, dtype=np.uint64)[0]
    res_idx, cnts = _sample_phase(mask, k, TR, seeds)

    # ---- 串行 merge：用 numpy 原生随机，等价于原版的顺序超几何分解 ----
    rng = np.random.RandomState(seed + 0x0D000721)
    total = int(cnts.sum())
    n_out = min(k, total)
    final_idx = np.empty(n_out, dtype=np.int64)
    if n_out == 0:
        return final_idx
    rem_sel = n_out
    rem_pop = total
    out = 0
    for t in range(TR):
        c_t = int(cnts[t])
        if c_t == 0 or rem_sel == 0:
            continue
        m_t = min(k, c_t)
        if t == TR - 1 or rem_pop == c_t:
            n_t = rem_sel
        else:
            n_t = int(rng.hypergeometric(c_t, rem_pop - c_t, rem_sel))
        n_t = min(n_t, m_t)
        idx_buf = np.arange(m_t)
        for j in range(n_t):
            r = j + int(rng.randint(0, m_t - j))
            idx_buf[j], idx_buf[r] = idx_buf[r], idx_buf[j]
        final_idx[out : out + n_t] = res_idx[t, idx_buf[:n_t]]
        out += n_t
        rem_sel -= n_t
        rem_pop -= c_t
    rng.shuffle(final_idx)
    return final_idx


def find_first_true(arr):
    """返回第一个 True 的下标，没有就返回 -1。arr 可以是 numpy 或 awkward 数组。

    改用 numpy 而不是 njit 串行循环：4.45M 的 bool 上 np.argmax 全扫一遍
    0.07ms、命中 0.04ms，而等价的 njit 循环要 1.13ms / 0.76ms（约 16 倍），
    因为 numpy 用的是向量化的扫描循环。真实调用形态（awkward & awkward 的
    结果）上是 0.12ms vs 1.64ms（约 13 倍）。
    """
    a = np.asarray(arr)
    if a.dtype != np.bool_:
        a = a.astype(np.bool_)
    if a.size == 0:
        return -1
    i = int(np.argmax(a))
    return i if a[i] else -1


@njit(cache=True)
def find_first_equal_where_index_greater(eqarr, gtpos, eqval):  # sourcery skip: use-next
    for i in range(gtpos + 1, len(eqarr)):
        if eqarr[i] == eqval:
            return i
    return -1


get_k_ts(np.array([], dtype=np.bool_), 1, 1, 0)
find_first_true(np.array([], dtype=np.bool_))

for t1, t2 in product([np.uint16, np.uint8], repeat=2):
    find_first_equal_where_index_greater(np.array([], dtype=t1), 0, 0)

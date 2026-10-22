import numpy as np

if __package__:
    from .queryerrors import QuerySyntaxError, QueryTooComplexError
else:
    from queryerrors import QuerySyntaxError, QueryTooComplexError  # type: ignore


class NFA:
    # 通用Thompson NFA构造器：token边消费一个符号，ε边表示选择、可选和重复。
    # 符号本身由整数掩码对(and, xand)描述，匹配语义由使用方决定；
    # 量词按需重复factory，不展开成额外状态时才用star。
    # arrays()按CSR布局序列化，便于numba内核逐状态扫描。

    def __init__(self, max_states: int = 8192, max_tokens: int = 4096):
        self.n = 0
        self.eps: list[tuple[int, int]] = []
        self.toks: list[tuple[int, int, int]] = []
        self.masks: list[tuple[int, int]] = []
        self._mask_idx: dict[tuple[int, int], int] = {}
        self.max_states = max_states
        self.max_tokens = max_tokens
        self.tok_copies = 0

    def state(self) -> int:
        s = self.n
        self.n += 1
        if self.n > self.max_states:
            raise QueryTooComplexError(f"查询过于复杂（状态数上限 {self.max_states}）")
        return s

    def add_mask(self, and_val: int, xand_val: int) -> int:
        key = (and_val, xand_val)
        idx = self._mask_idx.get(key)
        if idx is None:
            idx = len(self.masks)
            self._mask_idx[key] = idx
            self.masks.append(key)
        return idx

    def eps_edge(self, a: int, b: int):
        self.eps.append((a, b))

    def token_edge(self, a: int, b: int, mask: int):
        self.tok_copies += 1
        if self.tok_copies > self.max_tokens:
            raise QueryTooComplexError(f"查询过于复杂（token数上限 {self.max_tokens}）")
        self.toks.append((a, b, mask))

    def empty(self) -> tuple[int, int]:
        s = self.state()
        return s, s

    def token(self, mask: int) -> tuple[int, int]:
        a = self.state()
        b = self.state()
        self.token_edge(a, b, mask)
        return a, b

    def seq(self, frags: list[tuple[int, int]]) -> tuple[int, int]:
        if not frags:
            return self.empty()
        start, prev = frags[0]
        for s, a in frags[1:]:
            self.eps_edge(prev, s)
            prev = a
        return start, prev

    def alt(self, frags: list[tuple[int, int]]) -> tuple[int, int]:
        if len(frags) == 1:
            return frags[0]
        s = self.state()
        a = self.state()
        for fs, fa in frags:
            self.eps_edge(s, fs)
            self.eps_edge(fa, a)
        return s, a

    def star(self, factory) -> tuple[int, int]:
        s = self.state()
        a = self.state()
        self.eps_edge(s, a)
        fs, fa = factory()
        self.eps_edge(s, fs)
        self.eps_edge(fa, fs)
        self.eps_edge(fa, a)
        return s, a

    def repeat(self, factory, lo: int, hi: int | None) -> tuple[int, int]:
        frags = [factory() for _ in range(lo)]
        if hi is None:
            frags.append(self.star(factory))
        else:
            if hi < lo:
                raise QuerySyntaxError(f"量词范围非法: {{{lo},{hi}}}")
            for _ in range(hi - lo):
                fs, fa = factory()
                self.eps_edge(fs, fa)  # 可跳过这一份
                frags.append((fs, fa))
        return self.seq(frags)

    def arrays(self):
        n = self.n
        eps_off = np.zeros(n + 1, dtype=np.int64)
        for s, _ in self.eps:
            eps_off[s + 1] += 1
        np.cumsum(eps_off, out=eps_off)
        eps_to = np.zeros(len(self.eps), dtype=np.int64)
        pos = eps_off[:-1].copy()
        for s, d in self.eps:
            eps_to[pos[s]] = d
            pos[s] += 1

        tok_off = np.zeros(n + 1, dtype=np.int64)
        for s, _, _ in self.toks:
            tok_off[s + 1] += 1
        np.cumsum(tok_off, out=tok_off)
        tok_to = np.zeros(len(self.toks), dtype=np.int64)
        tok_mask = np.zeros(len(self.toks), dtype=np.int64)
        pos = tok_off[:-1].copy()
        for s, d, m in self.toks:
            tok_to[pos[s]] = d
            tok_mask[pos[s]] = m
            pos[s] += 1

        and16 = np.array([m[0] for m in self.masks], dtype=np.uint16)
        xand16 = np.array([m[1] for m in self.masks], dtype=np.uint16)
        return eps_off, eps_to, tok_off, tok_to, tok_mask, and16, xand16

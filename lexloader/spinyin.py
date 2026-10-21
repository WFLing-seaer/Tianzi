import logging
import re
from typing import NamedTuple

import numpy as np
import pinyinparser

if __package__:
    from .colproto import Pinyin
    from .typing_utils import ArrayLike
else:
    from colproto import Pinyin  # type: ignore
    from typing_utils import ArrayLike  # type: ignore

logger = logging.getLogger(__name__)

# SPinyin查询串的语法与编译。SPinyin类本身留在schemas.py，
# 这里只放与schema框架无关、可单独复用的部分。

PPINST = pinyinparser.Parser(
    pinyinparser.TOKENS.BASIC | pinyinparser.TOKENS.NE | pinyinparser.TOKENS.EXT | pinyinparser.TOKENS.EXT_NE | pinyinparser.TOKENS.EXT2
)

WCSPEC_DCT = {m.name: m.value for m in pinyinparser.Initial if m.name not in {"missing", "nul", "unspec"}} | {
    m.name: m.value for m in pinyinparser.Final if m.name not in {"missing", "nul", "unspec"}
}

TOKENS_PAT = "|".join(map(re.escape, sorted(WCSPEC_DCT.keys(), key=len, reverse=True)))

WCPAIR_PAT_S = f"([.?/]{{2,3}})({TOKENS_PAT})"

WCPAIR_PAT = re.compile(WCPAIR_PAT_S)

QUANTIFIERS = {"*": (0, None), "+": (1, None)}
# SQL LIKE 风格的通配：_ 是任意一个音节，% 是任意个（含0个）音节，二者本身无需前置音节
LIKE_WILDCARDS = {"_": (1, 1), "%": (0, None)}
# 有界量词 {n} / {n,} / {n,m}
BRACE_QUANT_PAT = re.compile(r"\{([0-9]+)(?:,([0-9]*))?\}")
# 展开(...)和[...]时的分支上限，避免组合数爆炸
MAX_EXPAND = 256

_INITIAL_ABSENT = {int(pinyinparser.Initial.missing), int(pinyinparser.Initial.unspec), int(pinyinparser.Initial.nul)}
_FINAL_ABSENT = {int(pinyinparser.Final.missing), int(pinyinparser.Final.unspec), int(pinyinparser.Final.nul)}
_TONE_ABSENT = {int(pinyinparser.Tone.missing), int(pinyinparser.Tone.unspec), int(pinyinparser.Tone.nul)}


def char_kind(ch: str) -> str | None:
    # 字符类中的字符是在同一文本位置上互相替换的，必须扮演同一种角色。
    # 例如 [b-d] 都是声母，而 [a-c] 混用了韵母和声母，不可能都合法。
    if ch in "012345":
        return "声调"
    try:
        part = PPINST.parse(ch)
    except ValueError:
        return None
    if len(part) != 1:
        return None
    syl = part[0]
    has_i = int(syl.initial) not in _INITIAL_ABSENT
    has_f = int(syl.final) not in _FINAL_ABSENT
    has_t = int(syl.tone) not in _TONE_ABSENT
    if has_i and has_f:
        return "音节"
    if has_i:
        return "声母"
    if has_f:
        return "韵母"
    if has_t:
        return "声调"
    return None


class _PText(NamedTuple):
    # 一段连续的拼音字面量（未带量词的括号/字符类已合并进来）
    text: str


class _PGroup(NamedTuple):
    # 带量词的括号组/字符类。alts是若干“节点序列”，rep作用于整组。
    alts: list
    rep: tuple[int, int | None]


class _NFA:
    # Thompson NFA构造器：token边消费一个目标音节，ε边表示选择、可选和重复。
    # 音节级量词仍然放在token边自己的repeat里，不展开成多个状态；
    # 只有括号组量词才需要NFA来表达“整组重复”。

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
            raise ValueError(f"拼音查询过于复杂（状态数上限 {self.max_states}）")
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
            raise ValueError(f"拼音查询过于复杂（token数上限 {self.max_tokens}）")
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
                raise ValueError(f"拼音量词范围非法: {{{lo},{hi}}}")
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


def parse_pinyin_pattern(s: str) -> list[list]:
    # 解析为若干“节点序列”变体。
    # 未带量词的括号/字符类先按字面展开并与相邻字面量合并，保持
    # (zh|w)(un|ang) -> zhun/zhang/wun/wang 的旧语义；
    # 带量词的括号/字符类保留为_PGroup，量词作用于整组。
    variants, pos = parse_pat_seq(s, 0, "")
    if pos != len(s):
        raise ValueError(f"不合法的拼音查询 @ {s[pos:]!r}")
    uniq: list[list] = []
    seen: set[tuple] = set()
    for v in variants:
        key = pat_key(v)
        if key not in seen:
            seen.add(key)
            uniq.append(v)
    return uniq


def pat_key(pattern: list) -> tuple:
    return tuple(
        ("t", n.text) if isinstance(n, _PText) else ("g", tuple(pat_key(a) for a in n.alts), n.rep) for n in pattern
    )


def parse_pat_seq(s: str, pos: int, stops: str) -> tuple[list[list], int]:
    variants: list[list] = [[]]
    while pos < len(s):
        ch = s[pos]
        if ch in stops:
            return variants, pos
        if ch == "(":
            alts, pos = parse_pat_alt(s, pos + 1)
            rep, npos = parse_pat_quant(s, pos)
            if rep is None:
                variants = [cat(v, a) for v in variants for a in alts]
            else:
                variants = [v + [_PGroup(alts, rep)] for v in variants]
                pos = npos
        elif ch == "[":
            chars, pos = expand_class(s, pos + 1)
            rep, npos = parse_pat_quant(s, pos)
            if rep is None:
                variants = [append_text(v, c) for v in variants for c in chars]
            else:
                alts = [[_PText(c)] for c in chars]
                variants = [v + [_PGroup(alts, rep)] for v in variants]
                pos = npos
        elif ch in ")]|":
            raise ValueError(f"不匹配的 {ch!r}")
        else:
            variants = [append_text(v, ch) for v in variants]
            pos += 1
        if len(variants) > MAX_EXPAND:
            raise ValueError(f"拼音查询的分支过多（上限 {MAX_EXPAND}）")
    if stops:
        raise ValueError("未闭合的 '('")
    return variants, pos


def parse_pat_alt(s: str, pos: int) -> tuple[list[list], int]:
    alts, pos = parse_pat_seq(s, pos, "|)")
    while pos < len(s) and s[pos] == "|":
        seq, pos = parse_pat_seq(s, pos + 1, "|)")
        alts.extend(seq)
    if pos >= len(s):
        raise ValueError("未闭合的 '('")
    return alts, pos + 1


def parse_pat_quant(s: str, pos: int) -> tuple[tuple[int, int | None] | None, int]:
    # 只有紧跟 ')' 或 ']' 的量词才作用于整组；紧跟字面量的量词留在字面量里，
    # 由parse_pinyin_query按音节粒度处理。
    if pos >= len(s):
        return None, pos
    ch = s[pos]
    if ch in QUANTIFIERS:
        return QUANTIFIERS[ch], pos + 1
    if ch == "{":
        m = BRACE_QUANT_PAT.match(s, pos)
        if m is None:
            raise ValueError(f"不合法的拼音量词 @ {s[pos:]!r}")
        lo = int(m.group(1))
        if m.group(2) is None:
            rep = (lo, lo)
        elif not m.group(2):
            rep = (lo, None)
        else:
            hi = int(m.group(2))
            if lo > hi:
                raise ValueError(f"拼音量词范围非法: {m.group(0)!r}")
            rep = (lo, hi)
        return rep, m.end()
    return None, pos


def append_text(v: list, c: str) -> list:
    if v and isinstance(v[-1], _PText):
        return v[:-1] + [_PText(v[-1].text + c)]
    return v + [_PText(c)]


def cat(v: list, a: list) -> list:
    # 拼接两个节点序列；相邻字面量合并，使未量化括号仍按文本展开。
    if not a:
        return list(v)
    if not v:
        return list(a)
    if isinstance(v[-1], _PText) and isinstance(a[0], _PText):
        return v[:-1] + [_PText(v[-1].text + a[0].text)] + a[1:]
    return v + a


def expand_class(s: str, pos: int) -> tuple[list[str], int]:
    chars: list[str] = []
    while pos < len(s) and s[pos] != "]":
        if pos + 2 < len(s) and s[pos + 1] == "-" and s[pos + 2] != "]":
            lo, hi = s[pos], s[pos + 2]
            if ord(lo) > ord(hi):
                raise ValueError(f"非法的字符范围 {lo}-{hi}")
            chars.extend(chr(c) for c in range(ord(lo), ord(hi) + 1))
            pos += 3
        else:
            chars.append(s[pos])
            pos += 1
    if pos >= len(s):
        raise ValueError("未闭合的 '['")
    if not chars:
        raise ValueError("空的字符类 []")
    chars = list(dict.fromkeys(chars))
    kinds: dict[str, str] = {}
    for ch in chars:
        kind = char_kind(ch)
        if kind is None:
            raise ValueError(f"字符类中的 {ch!r} 不是合法的拼音片段")
        kinds[ch] = kind
    first = chars[0]
    for ch in chars[1:]:
        if kinds[ch] != kinds[first]:
            raise ValueError(
                f"字符类中的字符必须属于同一类（声母/韵母/声调）: {first!r} 是{kinds[first]}，{ch!r} 是{kinds[ch]}，不能出现在同一位置"
            )
    return chars, pos + 1


def parse_pinyin_query(s: str, ppinst=PPINST) -> tuple[list[pinyinparser.Syllable], list[tuple[int, int | None]]]:
    # 量词以音节为粒度：* 任意次、+ 至少一次、{n}/{n,}/{n,m} 有界重复。
    # . ? / 保留给开头的声韵调spec，任意音节用 SQL LIKE 风格的 _ 和 %。
    sylls: list[pinyinparser.Syllable] = []
    reps: list[tuple[int, int | None]] = []
    buf: list[str] = []

    def flush():
        if not buf:
            return
        part = ppinst.parse("".join(buf))
        buf.clear()
        for syl in part:
            sylls.append(syl)
            reps.append((1, 1))

    def quantify(rep: tuple[int, int | None], text: str):
        if not reps:
            raise ValueError(f"拼音量词 {text!r} 前没有可重复的音节")
        if reps[-1] != (1, 1):
            raise ValueError(f"拼音量词 {text!r} 不能重复作用于同一个音节")
        reps[-1] = rep

    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "' -":
            flush()
            i += 1
        elif ch in LIKE_WILDCARDS:
            flush()
            reps.append(LIKE_WILDCARDS[ch])
            sylls.append(ppinst.parse(".")[0])  # 任意音节
            i += 1
        elif ch in QUANTIFIERS:
            flush()
            quantify(QUANTIFIERS[ch], ch)
            i += 1
        elif ch == "{":
            m = BRACE_QUANT_PAT.match(s, i)
            if m is None:
                raise ValueError(f"不合法的拼音量词 @ {s[i:]!r}")
            lo = int(m.group(1))
            if m.group(2) is None:
                rep = (lo, lo)
            elif not m.group(2):
                rep = (lo, None)
            else:
                hi = int(m.group(2))
                if lo > hi:
                    raise ValueError(f"拼音量词范围非法: {m.group(0)!r}")
                rep = (lo, hi)
            flush()
            quantify(rep, m.group(0))
            i = m.end()
        else:
            buf.append(ch)
            i += 1
    flush()
    return sylls, reps


def spec_flags(
    pinyins: list[pinyinparser.Syllable],
    wcspec: str,
    wcspecp: str,
    wc_pairs: list[tuple[str, str]],
):
    if not wcspec and not wcspecp:
        return False, False, False

    iws: list[bool] = []
    fws: list[bool] = []
    tws: list[bool] = []
    for syl in pinyins:
        wc = wcspec
        syl_i = int(syl.initial)
        syl_f = int(syl.final)
        for w, tk in wc_pairs:
            tv = WCSPEC_DCT.get(tk)
            if tv is not None and (syl_i == tv or syl_f == tv):
                wc = w
                break
        if not wc:
            wc = "..."

        iws.append(wc[0] == "?")
        if wc[0] == "/":
            syl.initial = pinyinparser.Initial.unspec
        fws.append(wc[1] == "?")
        if wc[1] == "/":
            syl.final = pinyinparser.Final.unspec
        tws.append(wc[2] == "?")
        if wc[2] == "/":
            syl.tone = pinyinparser.Tone.unspec

    logger.info(f"dbg: SPinyin {iws=} {fws=} {pinyins=}")
    return iws, fws, tws


def masks_for(
    pinyinc: Pinyin,
    pinyins: list[pinyinparser.Syllable],
    wcspec: str,
    wcspecp: str,
    wc_pairs: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    iws, fws, tws = spec_flags(pinyins, wcspec, wcspecp, wc_pairs)
    return pinyinc._build_masks(iws, fws, tws, pinyins)


def query_parsed(
    pinyinc: Pinyin,
    pinyins: list[pinyinparser.Syllable],
    reps: list[tuple[int, int | None]] | None,
    istart: int | None,
    iend: int | None,
    wcspec: str,
    wcspecp: str,
    wc_pairs: list[tuple[str, str]],
) -> ArrayLike:
    use_re = reps is not None and any(rep != (1, 1) for rep in reps)

    if not wcspec and not wcspecp:
        if use_re:
            return pinyinc.query_re(istart, iend, False, False, False, pinyins, reps)  # type: ignore[arg-type]
        return pinyinc.query(istart, iend, False, False, False, pinyins)

    iws, fws, tws = spec_flags(pinyins, wcspec, wcspecp, wc_pairs)
    if use_re:
        return pinyinc.query_re(istart, iend, iws, fws, tws, pinyins, reps)  # type: ignore[arg-type]
    return pinyinc.query(istart, iend, iws, fws, tws, pinyins)


def compile_nodes(
    nfa: _NFA,
    pinyinc: Pinyin,
    nodes: list,
    wcspec: str,
    wcspecp: str,
    wc_pairs: list[tuple[str, str]],
    ppinst=PPINST,
) -> tuple[int, int]:
    frags: list[tuple[int, int]] = []
    for node in nodes:
        if isinstance(node, _PText):
            sylls, reps = parse_pinyin_query(node.text, ppinst)
            and_np, xand_np = masks_for(pinyinc, sylls, wcspec, wcspecp, wc_pairs)
            for (lo, hi), and_val, xand_val in zip(reps, and_np, xand_np):
                mid = nfa.add_mask(int(and_val), int(xand_val))
                frags.append(nfa.repeat(lambda mid=mid: nfa.token(mid), lo, hi))
        else:
            assert isinstance(node, _PGroup)

            def factory(alts=node.alts):
                return nfa.alt(
                    [compile_nodes(nfa, pinyinc, alt, wcspec, wcspecp, wc_pairs, ppinst) for alt in alts]
                )

            frags.append(nfa.repeat(factory, node.rep[0], node.rep[1]))
    return nfa.seq(frags)

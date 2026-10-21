import logging
import re
from typing import NamedTuple

import numpy as np
import pinyinparser
import pypinyin
import pypinyin_dict.phrase_pinyin_data.large_pinyin
import pypinyin_dict.pinyin_data.cc_cedict

if __package__:
    from .colproto import Pinyin
    from .schema_base import SchemaABC, schema
    from .typing_utils import ArrayLike
else:
    from colproto import Pinyin  # type: ignore
    from schema_base import SchemaABC, schema  # type: ignore
    from typing_utils import ArrayLike  # type: ignore

pypinyin_dict.phrase_pinyin_data.large_pinyin.load()
pypinyin_dict.pinyin_data.cc_cedict.load()

logger = logging.getLogger(__name__)


# region SPinyin

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


@schema
class SPinyin(SchemaABC):
    # pinyin可以jit但是建议aot
    class required_cols(NamedTuple):
        pinyin: Pinyin

    cols: required_cols

    import pinyinparser

    ppinst = pinyinparser.Parser(
        pinyinparser.TOKENS.BASIC | pinyinparser.TOKENS.NE | pinyinparser.TOKENS.EXT | pinyinparser.TOKENS.EXT_NE | pinyinparser.TOKENS.EXT2
    )

    wcspec_dct = {m.name: m.value for m in pinyinparser.Initial if m.name not in {"missing", "nul", "unspec"}} | {
        m.name: m.value for m in pinyinparser.Final if m.name not in {"missing", "nul", "unspec"}
    }

    _tokens_pat = "|".join(map(re.escape, sorted(wcspec_dct.keys(), key=len, reverse=True)))

    _wcpair_pat = re.compile(_wcpair_pat_s := f"([.?/]{{2,3}})({_tokens_pat})")

    query_re_pat = re.compile(
        rf":(((?:\[(?P<start>-?[0-9]*):(?P<end>-?[0-9]*)\])?(?P<wcspecp>({_wcpair_pat_s}(?=[.?/]{{2,3}}))*)(?P<wcspec>[.?/]{{2,3}})?((?P<as>~[^ ]+)|(?P<pinyin>(?![*+])[012345abcdefghijklmnopqrstuvwxyzàáèéêìíòóùúüāēěīńňōūǎǐǒǔǖǘǚǜǹ̀́̄̌ḿếề' ?*+\-_%()|\[\]{{}},]+))))"
    )

    _quantifiers = {"*": (0, None), "+": (1, None)}
    # SQL LIKE 风格的通配：_ 是任意一个音节，% 是任意个（含0个）音节，二者本身无需前置音节
    _like_wildcards = {"_": (1, 1), "%": (0, None)}
    # 有界量词 {n} / {n,} / {n,m}
    _brace_quant_pat = re.compile(r"\{([0-9]+)(?:,([0-9]*))?\}")
    # 展开(...)和[...]时的分支上限，避免组合数爆炸
    _max_expand = 256

    @classmethod
    def _parse_pinyin_pattern(cls, s: str) -> list[list]:
        # 解析为若干“节点序列”变体。
        # 未带量词的括号/字符类先按字面展开并与相邻字面量合并，保持
        # (zh|w)(un|ang) -> zhun/zhang/wun/wang 的旧语义；
        # 带量词的括号/字符类保留为_PGroup，量词作用于整组。
        variants, pos = cls._parse_pat_seq(s, 0, "")
        if pos != len(s):
            raise ValueError(f"不合法的拼音查询 @ {s[pos:]!r}")
        uniq: list[list] = []
        seen: set[tuple] = set()
        for v in variants:
            key = cls._pat_key(v)
            if key not in seen:
                seen.add(key)
                uniq.append(v)
        return uniq

    @classmethod
    def _pat_key(cls, pattern: list) -> tuple:
        return tuple(
            ("t", n.text) if isinstance(n, _PText) else ("g", tuple(cls._pat_key(a) for a in n.alts), n.rep)
            for n in pattern
        )

    @classmethod
    def _parse_pat_seq(cls, s: str, pos: int, stops: str) -> tuple[list[list], int]:
        variants: list[list] = [[]]
        while pos < len(s):
            ch = s[pos]
            if ch in stops:
                return variants, pos
            if ch == "(":
                alts, pos = cls._parse_pat_alt(s, pos + 1)
                rep, npos = cls._parse_pat_quant(s, pos)
                if rep is None:
                    variants = [cls._cat(v, a) for v in variants for a in alts]
                else:
                    variants = [v + [_PGroup(alts, rep)] for v in variants]
                    pos = npos
            elif ch == "[":
                chars, pos = cls._expand_class(s, pos + 1)
                rep, npos = cls._parse_pat_quant(s, pos)
                if rep is None:
                    variants = [cls._append_text(v, c) for v in variants for c in chars]
                else:
                    alts = [[_PText(c)] for c in chars]
                    variants = [v + [_PGroup(alts, rep)] for v in variants]
                    pos = npos
            elif ch in ")]|":
                raise ValueError(f"不匹配的 {ch!r}")
            else:
                variants = [cls._append_text(v, ch) for v in variants]
                pos += 1
            if len(variants) > cls._max_expand:
                raise ValueError(f"拼音查询的分支过多（上限 {cls._max_expand}）")
        if stops:
            raise ValueError("未闭合的 '('")
        return variants, pos

    @classmethod
    def _parse_pat_alt(cls, s: str, pos: int) -> tuple[list[list], int]:
        alts, pos = cls._parse_pat_seq(s, pos, "|)")
        while pos < len(s) and s[pos] == "|":
            seq, pos = cls._parse_pat_seq(s, pos + 1, "|)")
            alts.extend(seq)
        if pos >= len(s):
            raise ValueError("未闭合的 '('")
        return alts, pos + 1

    @classmethod
    def _parse_pat_quant(cls, s: str, pos: int) -> tuple[tuple[int, int | None] | None, int]:
        # 只有紧跟 ')' 或 ']' 的量词才作用于整组；紧跟字面量的量词留在字面量里，
        # 由_parse_pinyin_query按音节粒度处理。
        if pos >= len(s):
            return None, pos
        ch = s[pos]
        if ch in cls._quantifiers:
            return cls._quantifiers[ch], pos + 1
        if ch == "{":
            m = cls._brace_quant_pat.match(s, pos)
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

    @staticmethod
    def _append_text(v: list, c: str) -> list:
        if v and isinstance(v[-1], _PText):
            return v[:-1] + [_PText(v[-1].text + c)]
        return v + [_PText(c)]

    @classmethod
    def _cat(cls, v: list, a: list) -> list:
        # 拼接两个节点序列；相邻字面量合并，使未量化括号仍按文本展开。
        if not a:
            return list(v)
        if not v:
            return list(a)
        if isinstance(v[-1], _PText) and isinstance(a[0], _PText):
            return v[:-1] + [_PText(v[-1].text + a[0].text)] + a[1:]
        return v + a

    @classmethod
    def _expand_class(cls, s: str, pos: int) -> tuple[list[str], int]:
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
        return list(dict.fromkeys(chars)), pos + 1

    @classmethod
    def _parse_pinyin_query(cls, s: str) -> tuple[list[pinyinparser.Syllable], list[tuple[int, int | None]]]:
        # 量词以音节为粒度：* 任意次、+ 至少一次、{n}/{n,}/{n,m} 有界重复。
        # . ? / 保留给开头的声韵调spec，任意音节用 SQL LIKE 风格的 _ 和 %。
        sylls: list[pinyinparser.Syllable] = []
        reps: list[tuple[int, int | None]] = []
        buf: list[str] = []

        def flush():
            if not buf:
                return
            part = cls.ppinst.parse("".join(buf))
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
            elif ch in cls._like_wildcards:
                flush()
                reps.append(cls._like_wildcards[ch])
                sylls.append(cls.ppinst.parse(".")[0])  # 任意音节
                i += 1
            elif ch in cls._quantifiers:
                flush()
                quantify(cls._quantifiers[ch], ch)
                i += 1
            elif ch == "{":
                m = cls._brace_quant_pat.match(s, i)
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

    def _spec_flags(
        self,
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
                tv = self.wcspec_dct.get(tk)
                if tv is not None and (syl_i == tv or syl_f == tv):
                    wc = w
                    break
            if not wc:
                wc = "..."

            iws.append(wc[0] == "?")
            if wc[0] == "/":
                syl.initial = self.pinyinparser.Initial.unspec
            fws.append(wc[1] == "?")
            if wc[1] == "/":
                syl.final = self.pinyinparser.Final.unspec
            tws.append(wc[2] == "?")
            if wc[2] == "/":
                syl.tone = self.pinyinparser.Tone.unspec

        logger.info(f"dbg: SPinyin {iws=} {fws=} {pinyins=}")
        return iws, fws, tws

    def _masks_for(
        self,
        pinyinc: Pinyin,
        pinyins: list[pinyinparser.Syllable],
        wcspec: str,
        wcspecp: str,
        wc_pairs: list[tuple[str, str]],
    ) -> tuple[np.ndarray, np.ndarray]:
        iws, fws, tws = self._spec_flags(pinyins, wcspec, wcspecp, wc_pairs)
        return pinyinc._build_masks(iws, fws, tws, pinyins)

    def _query_parsed(
        self,
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

        iws, fws, tws = self._spec_flags(pinyins, wcspec, wcspecp, wc_pairs)
        if use_re:
            return pinyinc.query_re(istart, iend, iws, fws, tws, pinyins, reps)  # type: ignore[arg-type]
        return pinyinc.query(istart, iend, iws, fws, tws, pinyins)

    def _compile_nodes(
        self,
        nfa: _NFA,
        pinyinc: Pinyin,
        nodes: list,
        wcspec: str,
        wcspecp: str,
        wc_pairs: list[tuple[str, str]],
    ) -> tuple[int, int]:
        frags: list[tuple[int, int]] = []
        for node in nodes:
            if isinstance(node, _PText):
                sylls, reps = self._parse_pinyin_query(node.text)
                and_np, xand_np = self._masks_for(pinyinc, sylls, wcspec, wcspecp, wc_pairs)
                for (lo, hi), and_val, xand_val in zip(reps, and_np, xand_np):
                    mid = nfa.add_mask(int(and_val), int(xand_val))
                    frags.append(nfa.repeat(lambda mid=mid: nfa.token(mid), lo, hi))
            else:
                assert isinstance(node, _PGroup)

                def factory(alts=node.alts):
                    return nfa.alt(
                        [self._compile_nodes(nfa, pinyinc, alt, wcspec, wcspecp, wc_pairs) for alt in alts]
                    )

                frags.append(nfa.repeat(factory, node.rep[0], node.rep[1]))
        return nfa.seq(frags)

    def query(self, mch):
        pinyinc = self.cols.pinyin

        istart: int | None = (mch.group("start") or None) and int(mch.group("start"))
        iend: int | None = (mch.group("end") or None) and int(mch.group("end"))

        wcspecp = mch.group("wcspecp") or ""
        wcspec = mch.group("wcspec") or ""
        if wcspec and len(wcspec) == 2:
            wcspec += "."

        wc_pairs: list[tuple[str, str]] = self._wcpair_pat.findall(wcspecp)
        wc_pairs = [(f"{w}." if len(w) == 2 else w, tk) for w, tk in wc_pairs]
        logger.info(f"dbg: SPinyin {wcspec=} {wc_pairs=}")

        if as_ := mch.group("as"):
            pinyins = list(
                self.pinyinparser.parse(
                    "'".join(
                        pypinyin.lazy_pinyin(
                            as_,
                            style=pypinyin.Style.TONE,
                            neutral_tone_with_five=True,
                            errors=lambda _: None,  # type: ignore
                        )
                    )
                )
            )
            return self._query_parsed(pinyinc, pinyins, None, istart, iend, wcspec, wcspecp, wc_pairs)

        pinyin: str = mch.group("pinyin") or ""
        results: list[ArrayLike] = []
        for pattern in self._parse_pinyin_pattern(pinyin):
            if any(isinstance(node, _PGroup) for node in pattern):
                nfa = _NFA()
                start, accept = self._compile_nodes(nfa, pinyinc, pattern, wcspec, wcspecp, wc_pairs)
                eps_off, eps_to, tok_off, tok_to, tok_mask, and16, xand16 = nfa.arrays()
                results.append(
                    pinyinc.query_re_nfa(
                        istart,
                        iend,
                        and16,
                        xand16,
                        eps_off,
                        eps_to,
                        tok_off,
                        tok_to,
                        tok_mask,
                        start,
                        accept,
                    )
                )
            else:
                text = "".join(node.text for node in pattern if isinstance(node, _PText))
                pinyins, reps = self._parse_pinyin_query(text)
                results.append(
                    self._query_parsed(pinyinc, pinyins, reps, istart, iend, wcspec, wcspecp, wc_pairs)
                )

        result = results[0]
        for r in results[1:]:
            result = result | r
        return result



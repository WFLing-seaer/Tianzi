import inspect
import logging
import re
import time
import types
import typing
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from itertools import chain
from typing import NamedTuple, cast

import numba
import numpy as np
import pinyinparser
import pypinyin
import pypinyin_dict.phrase_pinyin_data.large_pinyin
import pypinyin_dict.pinyin_data.cc_cedict
import recheck
from cachetools import LRUCache, TTLCache, cached

pypinyin_dict.phrase_pinyin_data.large_pinyin.load()
pypinyin_dict.pinyin_data.cc_cedict.load()

if __package__:
    from .colproto import (
        ColProtoABC,
        Pinyin,
        PlainText,
        SortedColABC,
        _Enum,
        _Int,
        _Length,
    )
    from .headparser import TColSpec
    from .nputil import find_first_equal_where_index_greater, find_first_true, get_k_ts
    from .typing_utils import ArrayLike, AwkwardLike, asArrayLike
else:
    from colproto import (
        ColProtoABC,
        Pinyin,
        PlainText,
        SortedColABC,
        _Enum,
        _Int,
        _Length,
    )
    from headparser import TColSpec
    from nputil import find_first_equal_where_index_greater, find_first_true, get_k_ts
    from typing_utils import ArrayLike, AwkwardLike, asArrayLike

logger = logging.getLogger(__name__)


schemas = []


def schema(cls: type[SchemaABC]) -> type[SchemaABC]:
    schemas.append(cls)
    return cls


class Schemas:
    def __init__(self, data: dict[TColSpec, ColProtoABC], idx_cache_size=4096):
        self.cols = {k.name: v for k, v in data.items()}
        self.orig_heads = list(data.keys())
        self.batch_cache_size = idx_cache_size
        self.qcache: TTLCache[str, np.ndarray | None] = TTLCache[str, np.ndarray | None](65536 // idx_cache_size, 600)

        group_cols: defaultdict[str, list[tuple[int, set[str] | None, ColProtoABC, type[ColProtoABC]]]] = defaultdict(list)
        for idx, (spec, col) in enumerate(data.items()):
            col_cls = type(col)
            for gname, gschemas in spec.group.items():
                group_cols[gname].append((idx, gschemas, col, col_cls))

        @cached(LRUCache(256))
        def check_type(col_cls: type[ColProtoABC], type_cls: type[ColProtoABC]) -> bool:
            if issubclass(col_cls, SortedColABC) and col_cls.__name__.endswith("S"):
                regular_cls_name = col_cls.__name__[:-1]
                regular_cls = next((cls for cls in col_cls.__mro__ if cls.__name__ == regular_cls_name), None)
                if regular_cls is not None and check_type(regular_cls, type_cls):
                    return True
            if not type_cls.__name__.startswith("_"):
                return col_cls is type_cls
            return any(not sub.__name__.startswith("_") and not inspect.isabstract(sub) and col_cls is sub for sub in type_cls.__subclasses__())

        def getmatch(avail: list[tuple[int, ColProtoABC, type[ColProtoABC]]], used: set[int], target: type[ColProtoABC]):
            return next(((idx, col) for idx, col, col_cls in avail if idx not in used and check_type(col_cls, target)), None)

        def scmatch(schema_cls: type[SchemaABC], all_cols: list[tuple[int, set[str] | None, ColProtoABC, type[ColProtoABC]]]):
            schema_name = schema_cls.__name__

            visibles = [(idx, col, col_cls) for idx, allowed, col, col_cls in all_cols if allowed is None or schema_name in allowed]

            req, opt_req = schema_cls.get_fields_req()
            all_typ = set(req.values()) | set(opt_req.values())

            typ_count = Counter(t for _, _, col_cls in visibles for t in all_typ if check_type(col_cls, t))
            req_count = Counter(req.values())
            opt_count = Counter(opt_req.values())

            if any(typ_count[t] < req_count[t] or typ_count[t] > req_count[t] + opt_count[t] for t in all_typ):
                return None

            field_assign: dict[str, ColProtoABC | None] = {}
            used: set[int] = set()

            for fname, ftype in req.items():
                mch = getmatch(visibles, used, ftype)
                if mch is None:
                    return None
                idx, col = mch
                field_assign[fname] = col
                used.add(idx)

            for fname, ftype in opt_req.items():
                mch = getmatch(visibles, used, ftype)
                if mch is not None:
                    idx, col = mch
                    field_assign[fname] = col
                    used.add(idx)
                else:
                    field_assign[fname] = None

            if any(idx not in used and any(check_type(col_cls, t) for t in all_typ) for idx, _, col_cls in visibles):
                return None

            anno = schema_cls.required_cols.__annotations__
            cols_inst = cast(NamedTuple, schema_cls.required_cols(*(field_assign.get(fname) for fname in anno)))  # type: ignore 今日份动态构建炸类型检查1/1
            schema_instance = schema_cls(cols_inst)

            return schema_instance if schema_instance.checkinit() else None

        self.schemas = {gname: [s for s in (scmatch(sc, cols) for sc in schemas) if s is not None] for gname, cols in group_cols.items()}

        for gname, gschemas in self.schemas.items():
            if not gschemas:
                logger.info(f"组 {gname!r} : N/A")
                continue
            for s in gschemas:
                cols = ", ".join(f"{f}={type(getattr(s.cols, f)).__name__}" for f in s.cols._fields if getattr(s.cols, f) is not None)
                logger.info(f"组 {gname!r} : {type(s).__name__} ({cols})")

    def query(self, q: str) -> ArrayLike | bool:
        if not q:
            logger.info("<空query>")
            return True

        logger.info(f"query: {q!r}")
        q_t0 = time.perf_counter()

        def _split_or(s: str) -> list[str]:  # 这三个其实按理来说可以复用textutil，但是由于沟槽的相对导入我宁可重写一遍（似了
            parts = []
            depth = 0
            i = 0
            i0 = 0
            while i < len(s):
                c = s[i]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                elif depth == 0 and c == "|":
                    parts.append(s[i0:i])
                    i += 1
                    i0 = i
                    continue
                i += 1
            parts.append(s[i0:])
            return parts

        def _split_and(s: str) -> list[str]:
            parts = []
            depth = 0
            i = 0
            i0 = 0
            while i < len(s):
                c = s[i]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                elif depth == 0 and c == "&":
                    parts.append(s[i0:i])
                    i += 1
                    i0 = i
                    continue
                i += 1
            parts.append(s[i0:])
            return parts

        def _split_g(s: str) -> tuple[str, str | None]:
            depth = 0
            for i, c in enumerate(s):
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                elif c == "@" and depth == 0:
                    return s[:i], s[i + 1 :]
            return s, None

        # ----- deepseek写的 -----
        def _match_consume(s: str, group_name: str | None) -> ArrayLike | bool:
            schemas_list = self.schemas[group_name] if group_name else list(chain.from_iterable(self.schemas.values()))
            if not self.cols:
                return False
            data_len = len(next(iter(self.cols.values())).data)
            result = np.ones(data_len, dtype=bool)
            remaining = s
            logger.info(f"match_consume gname={group_name!r} schemas={[type(sc).__name__ for sc in schemas_list]}")
            while remaining:
                for schema_inst in schemas_list:
                    t0 = time.perf_counter()
                    m = schema_inst.match(remaining)
                    tm = (time.perf_counter() - t0) * 1000
                    if m is not None:
                        t1 = time.perf_counter()
                        part_result = schema_inst.query(m)
                        tq = (time.perf_counter() - t1) * 1000
                        result = result & part_result
                        remaining = remaining[m.end() :]
                        logger.info(
                            f"match_consume : {type(schema_inst).__name__} → {m.group(0)!r} gd={ {k: v for k, v in (m.groupdict() or {}).items() if v}!r}"
                            f"match_consume {tm:.2f}ms query {tq:.2f}ms → {remaining!r}"
                        )
                        break
                else:
                    raise SyntaxError(f"不合法的查询 @ {remaining}")
            logger.info(f"match_consume : gname={group_name!r} ok")
            return asArrayLike(result)

        # ---------------------------------------------------------------

        or_parts = _split_or(q)
        logger.info(f"split or {or_parts!r}")
        or_results = []

        for or_part in or_parts:
            and_queries = _split_and(or_part)
            logger.info(f"split and {or_part!r} → {and_queries!r}")
            and_results = []

            for query_str in and_queries:
                query_str = query_str.strip()
                if not query_str:
                    continue

                neg_query = query_str.startswith("!")
                query_str = query_str[1:] if neg_query else query_str
                query_content, group_name = _split_g(query_str)
                query_content = query_content[1:-1] if query_content.startswith("(") and query_content.endswith(")") else query_content
                if group_name is not None:
                    group_name = group_name.strip()

                logger.info(f"neg={neg_query} gname={group_name!r} content={query_content!r}")
                t0 = time.perf_counter()
                result = _match_consume(query_content, group_name)
                dt = (time.perf_counter() - t0) * 1000

                if neg_query:
                    result = (not result) if isinstance(result, bool) else ~result  # type: ignore 每日一骂啥比awkward 但凡写点类型注解我也不至于沦落到弄一堆arraylike之类的玩意然后type:ignore满天飞了

                logger.info(f"and seg ok {dt:.1f}ms")
                and_results.append(result)

            if any(x is False for x in and_results):  # in会触发__bool__然后抛错，谁知道为啥
                logger.info("and skip")
                continue  # 有False整坨为假那也不用再过后续了
            all_true = bool(and_results)
            and_results = [r for r in and_results if r is not True]  # True就没必要and了我说
            if and_results:
                or_result = and_results[0]
                for r in and_results[1:]:
                    or_result = or_result & r
                or_results.append(or_result)
            elif all_true:
                logger.info("and at")
                or_results.append(True)

        if any(x is True for x in or_results):
            logger.info(f"or skip qt={(time.perf_counter() - q_t0) * 1000:.1f}ms")
            return True
        or_results = [r for r in or_results if r is not False]

        if or_results:
            result = or_results[0]
            for r in or_results[1:]:
                result = result | r
            logger.info(f"or ok qt={(time.perf_counter() - q_t0) * 1000:.1f}ms")
            return result
        else:
            logger.info(f"or af qt={(time.perf_counter() - q_t0) * 1000:.1f}ms")
            return False

    def query_some(self, q: str) -> np.ndarray | None:
        if q in self.qcache:
            logger.info(f"qsome cache hit @ {q!r}")
            return self.qcache[q]
        t0 = time.perf_counter()
        qr = self.query(q)
        dt = (time.perf_counter() - t0) * 1000
        logger.info(f"qsome dt={dt:.1f}ms")
        if qr is False:
            self.qcache[q] = None
            logger.info(f"qsome cache miss @ {q!r}")
            return None
        if qr is True:
            clen = len(next(iter(self.cols.values())).data)
            if clen <= self.batch_cache_size:
                chosen = np.arange(clen, dtype=np.int64)
            else:
                chosen = np.random.default_rng().choice(clen, self.batch_cache_size, replace=False)
        else:
            chosen = get_k_ts(qr, self.batch_cache_size, numba.get_num_threads(), np.random.randint(0, 2147483647))
        if (lch := len(chosen)) > 0:
            has_more = lch >= self.batch_cache_size  # 如果<说明一共就这些再取也没意义了
            chosen.dtype = np.dtype(chosen.dtype.name, metadata={"has_more": has_more})  # type: ignore #依然动态炸检）
            self.qcache[q] = chosen
            logger.info(f"qsome ← {lch} ({has_more}) @ {q!r}")
        else:
            self.qcache[q] = None
            logger.info("qsome ← nul")
        return chosen

    def query_pop(self, q: str) -> int | None:
        if q not in self.qcache:
            self.query_some(q)
        qc = self.qcache[q]
        if qc is None:
            logger.info(f"qpop cached empty @ {q!r}")
            return None
        if qc.dtype.metadata and qc.dtype.metadata["has_more"]:
            ret = qc[-1]
            if len(qc) <= 1:
                del self.qcache[q]
                logger.info(f"qpop cache drain @ {q!r}")
            else:
                self.qcache[q] = qc[:-1]
                logger.info(f"qpop cache pop {ret} @ {q!r} lqc-1={len(qc) - 1}")
        else:
            ret = np.random.choice(qc)
            logger.info(f"qpop cache rand {ret} @ {q!r} lqc={len(qc)}")
        return ret

    def get_col(self, n: str | None = None) -> ColProtoABC:
        return self.cols[n] if n else next((c for c in self.cols.values() if isinstance(c, PlainText)), next(iter(self.cols.values())))


class SchemaABC(ABC):
    required_cols: type[NamedTuple]
    query_re_pat: re.Pattern[str] = re.compile("")
    group: str

    @classmethod
    def get_fields_req(cls) -> tuple[dict[str, type[ColProtoABC]], dict[str, type[ColProtoABC]]]:
        anno = cls.required_cols.__annotations__
        req: dict[str, type[ColProtoABC]] = {}
        opt_req: dict[str, type[ColProtoABC]] = {}
        for k, v in anno.items():
            match typing.get_args(v):
                case () if isinstance(v, type) and issubclass(v, ColProtoABC):
                    req[k] = v
                case (types.NoneType, t) | (t, types.NoneType) if isinstance(t, type) and issubclass(t, ColProtoABC):
                    opt_req[k] = t
                case _:
                    raise SyntaxError("required_cols的每一项都必须注解为一个ColProto（必需列）或ColProto|None=None（可选列）。")
        return req, opt_req

    def __init__(self, cols: NamedTuple):
        self.cols = cols

    def match(self, s: str) -> re.Match | None:
        return self.query_re_pat.match(s)

    def checkinit(self):
        # schema可以自定义checkinit检查初始化分配的列，返回False表示初始化不合法
        return True

    @abstractmethod
    def query(self, mch: re.Match) -> AwkwardLike: ...


# region Schema s


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


@schema
class SWordLength(SchemaABC):
    class required_cols(NamedTuple):
        word: PlainText | None = None
        length: _Length | None = None

    cols: required_cols

    query_re_pat = re.compile("=(?P<o>(<|>|<=|>=|\\!=))?(?P<l>[0-9]+)")

    def checkinit(self):
        return (self.cols.word is not None) or (self.cols.length is not None)  # 二者都是可选但二者至少有其一

    def query(self, mch):
        l = int(mch.group("l"))
        o = mch.group("o")
        if self.cols.length is not None:
            return self.cols.length.query({"<": "lt", ">": "gt", "<=": "le", ">=": "ge", "!=": "ne"}.get(o, "eq"), l)

        wordc = cast(PlainText, self.cols.word)
        match o:
            case "<":
                return wordc.query("length", (), {}) < l
            case ">":
                return wordc.query("length", (), {}) > l
            case "<=":
                return wordc.query("length", (), {}) <= l
            case ">=":
                return wordc.query("length", (), {}) >= l
            case "!=":
                return wordc.query("length", (), {}) != l
            case _:
                return wordc.query("length", (), {}) == l


@schema
class SPath(SchemaABC):
    class required_cols(NamedTuple):
        ID: _Int
        parent: _Int
        name: PlainText

    cols: required_cols

    query_re_pat = re.compile("/(?P<path>[^ ]+)/\\*(?P<getall>\\*)?")

    def query(self, mch):  # sourcery skip: extract-method
        path: list[str] = mch.group("path").split("/")
        getall = bool(mch.group("getall"))
        current_node_id = 0
        current_node_pos = -1
        for pt in path:
            nmask = self.cols.parent.query("eq", current_node_id) & self.cols.name.query("__eq__", (pt,), {})
            current_node_pos = find_first_true(nmask)
            if current_node_pos == -1:
                return False
            current_node_id = cast(int, self.cols.ID.data[current_node_pos])
        if getall:
            rsibling_pos = find_first_equal_where_index_greater(
                eqarr=self.cols.parent.data, eqval=self.cols.parent.data[current_node_pos], gtpos=current_node_id
            )
            if rsibling_pos == -1:
                parend_id = cast(int, self.cols.parent.data[current_node_pos])
                if parend_id == 0:
                    return self.cols.ID.data > current_node_id
                parent_pos = find_first_true(self.cols.ID.query("eq", parend_id))
                parent_rsibling_pos = find_first_equal_where_index_greater(
                    eqarr=self.cols.parent.data, eqval=self.cols.parent.data[parent_pos], gtpos=parend_id
                )
                if parent_rsibling_pos == -1:
                    return self.cols.ID.data > current_node_id
                return (self.cols.ID.data < self.cols.ID.data[parent_rsibling_pos]) & (self.cols.ID.data > current_node_id)
            rsibling_id = cast(int, self.cols.ID.data[rsibling_pos])
            return (self.cols.ID.data < rsibling_id) & (self.cols.ID.data > current_node_id)
        return self.cols.parent.query("eq", current_node_id)


@schema
class SEnum(SchemaABC):
    class required_cols(NamedTuple):
        enum: _Enum

    cols: required_cols

    query_re_pat = re.compile("(?P<exclude>\\-?)#(?P<enum>[^ ]+)#")

    def query(self, mch):
        req = mch.group("enum")
        exclude = bool(mch.group("exclude"))
        return self.cols.enum.query("ne" if exclude else "eq", req)


@schema
class SPos(SchemaABC):
    class required_cols(NamedTuple):
        word: PlainText

    cols: required_cols

    query_re_pat = re.compile("#(?P<start>[0-9]+)?\\.\\.(?P<stop>[0-9]+)?")

    def query(self, mch):
        _start = mch.group("start")
        _stop = mch.group("stop")

        start = int(_start) if _start else None
        stop = int(_stop) if _stop else None

        ret = np.zeros(len(self.cols.word.data), dtype=np.bool_)
        ret[start:stop] = True

        return ret


@schema
class SRegex(SchemaABC):
    class required_cols(NamedTuple):
        word: PlainText

    cols: required_cols

    query_re_pat = re.compile("/(?P<regex>[^ ]+)/")

    rechecker = recheck.Rechecker()

    def query(self, mch):
        regex = mch.group("regex")
        if (check_ret := self.rechecker.check(regex)) > (recheck.Complexity.POLYNOMIAL, 2):
            raise MemoryError(f"正则过于复杂（{check_ret}）。")
        return self.cols.word.query("match_substring_regex", (f"^{regex.removeprefix("^").removesuffix("$")}$",), {})

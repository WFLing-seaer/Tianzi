import inspect
import re
import types
import typing
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from itertools import chain
from typing import NamedTuple, cast

import numba
import numpy as np
from cachetools import LRUCache, TTLCache, cached

if __package__:
    from .colproto import ColProtoABC, Pinyin, PlainText, SortedColABC, _Int, _Length
    from .headparser import TColSpec
    from .nputil import find_first_true, get_k_ts
    from .typing_utils import ArrayLike, AwkwardLike
else:
    from colproto import ColProtoABC, Pinyin, PlainText, SortedColABC, _Int, _Length
    from headparser import TColSpec
    from nputil import find_first_true, get_k_ts
    from typing_utils import ArrayLike, AwkwardLike


schemas = []


def schema(cls: type[SchemaABC]) -> type[SchemaABC]:
    schemas.append(cls)
    return cls


class Schemas:
    def __init__(self, data: dict[TColSpec, ColProtoABC], idx_cache_size=256):
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

    def query(self, q: str) -> ArrayLike | bool:
        if not q:
            return True

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

        or_parts = _split_or(q)
        or_results = []

        for or_part in or_parts:
            and_queries = _split_and(or_part)
            and_results = []
            is_first = True

            for query_str in and_queries:
                query_str = query_str.strip()
                if not query_str:
                    continue

                query_content, group_name = _split_g(query_str)
                query_content = query_content[1:-1] if query_content.startswith("(") and query_content.endswith(")") else query_content
                if group_name is not None:
                    group_name = group_name.strip()

                if is_first:
                    is_first = False
                    if group_name is None:
                        schemas_iter = chain.from_iterable(self.schemas.values())
                    else:
                        if group_name not in self.schemas:
                            raise KeyError(group_name)
                        schemas_iter = iter(self.schemas[group_name])
                else:
                    if group_name is None:
                        raise SyntaxError(f"除每个或组的第一个查询之外其余查询都必须指定组 @ {query_content}")
                    if group_name not in self.schemas:
                        raise KeyError(group_name)
                    schemas_iter = iter(self.schemas[group_name])

                matched = False
                seen_types = set()
                for schema in schemas_iter:
                    schema_type = type(schema)
                    if schema_type in seen_types:
                        continue
                    seen_types.add(schema_type)
                    result = schema.query(query_content)
                    if result is not None:
                        and_results.append(result)
                        matched = True

                if not matched:
                    raise SyntaxError(f"不合法的查询（无法被任何Schema匹配） @ {query_content}")

            if and_results:
                or_result = and_results[0]
                for r in and_results[1:]:
                    or_result = or_result & r
                or_results.append(or_result)

        if or_results:
            result = or_results[0]
            for r in or_results[1:]:
                result = result | r
            return result
        else:
            return False

    def query_some(self, q: str) -> np.ndarray | None:
        if q in self.qcache:
            return self.qcache[q]
        qr = self.query(q)
        if qr is False:
            self.qcache[q] = None
            return None
        if qr is True:
            chosen = np.random.randint(0, len(next(iter(self.cols.values())).data) - 1, self.batch_cache_size)
        else:
            chosen = get_k_ts(qr, self.batch_cache_size, numba.get_num_threads(), np.random.randint(0, 2147483647))
        print("debug: chosen", chosen)
        if (lch := len(chosen)) > 0:
            has_more = lch >= self.batch_cache_size  # 如果<说明一共就这些再取也没意义了
            chosen.dtype = np.dtype(chosen.dtype.name, metadata={"has_more": has_more})  # type: ignore #依然动态炸检）
            self.qcache[q] = chosen
        else:
            self.qcache[q] = None
        return chosen

    def query_pop(self, q: str) -> int | None:
        if q not in self.qcache:
            self.query_some(q)
        qc = self.qcache[q]
        print("debug: qc direct:", qc)
        if qc is None:
            return None
        if qc.dtype.metadata and qc.dtype.metadata["has_more"]:
            print("debug: pop")
            ret = qc[-1]
            if len(qc) <= 1:
                del self.qcache[q]
            else:
                self.qcache[q] = qc[:-1]
        else:
            print("debug: keep")
            ret = np.random.choice(qc)
        print("debug: pk ret", ret, qc)
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

    def query(self, q: str) -> AwkwardLike | None:
        mch = self.query_re_pat.search(q)
        return None if mch is None else self._query(mch)

    def checkinit(self):
        # schema可以自定义checkinit检查初始化分配的列，返回False表示初始化不合法
        return True

    @abstractmethod
    def _query(self, mch: re.Match) -> AwkwardLike: ...


# region Schema s


@schema
class SPinyin(SchemaABC):
    # pinyin可以jit但是建议aot
    class required_cols(NamedTuple):
        pinyin: Pinyin

    cols: required_cols

    import pinyinparser

    ppinst = pinyinparser.Parser(
        pinyinparser.TOKENS.BASIC
        | pinyinparser.TOKENS.NE
        | pinyinparser.TOKENS.EXT
        | pinyinparser.TOKENS.EXT_NE
        | {"?": [0x0001, 0x0100, 0x0020]}
    )

    wcspec_dct = {m.name: m.value for m in pinyinparser.Initial if m.name not in {"missing", "nul", "unspec"}} | {
        m.name: m.value for m in pinyinparser.Final if m.name not in {"missing", "nul", "unspec"}
    }

    _tokens_pat = "|".join(map(re.escape, sorted(wcspec_dct.keys(), key=len, reverse=True)))

    _wcpair_pat = re.compile(__wcpair_pat_s := f"([.?/]{{2}})({_tokens_pat})")

    query_re_pat = re.compile(
        rf":(?:\[(?P<start>-?[0-9]*):(?P<end>-?[0-9]*)\])?(?P<wcspecp>({__wcpair_pat_s}(?=[.?/]{{2}}))*)(?P<wcspec>[.?/]{{2}})?(?P<pinyin>[12345abcdefghijklmnopqrstuvwxyzàáèéêìíòóùúüāēěīńňōūǎǐǒǔǖǘǚǜǹ̀́̄̌ḿếề'?]+)"
    )

    def _query(self, mch):
        pinyin: str = mch.group("pinyin") or ""
        istart: int | None = (mch.group("start") or None) and int(mch.group("start"))
        iend: int | None = (mch.group("end") or None) and int(mch.group("end"))

        wcspecp = mch.group("wcspecp") or ""
        wcspec = mch.group("wcspec") or ""

        pinyins = self.ppinst.parse(pinyin)
        pinyinc = self.cols.pinyin

        if not wcspec and not wcspecp:
            return pinyinc.query(istart, iend, False, False, pinyins)

        wc_pairs: list[tuple[str, str]] = self._wcpair_pat.findall(wcspecp)

        print(f"debug {wcspec=} {wc_pairs=}")

        iws: list[bool] = []
        fws: list[bool] = []
        for syl in pinyins:
            wc = wcspec
            syl_i = int(syl.initial)
            syl_f = int(syl.final)
            for w, tk in wc_pairs:
                tv = self.wcspec_dct.get(tk)
                if tv is not None and (syl_i == tv or syl_f == tv):
                    wc = w
                    break
            iws.append(wc[0] == "?")
            fws.append(wc[1] == "?")
            if wc[0] == "/":
                syl.initial = self.pinyinparser.Initial.unspec
            if wc[1] == "/":
                syl.final = self.pinyinparser.Final.unspec

        print(f"debug: {iws=} {fws=} {pinyins=}")
        return pinyinc.query(istart, iend, iws, fws, pinyins)


@schema
class SWordLength(SchemaABC):
    class required_cols(NamedTuple):
        word: PlainText | None = None
        length: _Length | None = None

    cols: required_cols

    query_re_pat = re.compile("=(?P<o>(<|>|<=|>=|\\!=))?(?P<l>[0-9]+)")

    def checkinit(self):
        return (self.cols.word is not None) or (self.cols.length is not None)  # 二者都是可选但二者至少有其一

    def _query(self, mch):
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

    query_re_pat = re.compile("/(?P<path>[^ ]+)/\\*")

    def _query(self, mch):
        path = mch.group("path").split("/")
        if not any(path):
            return self.cols.parent.query("eq", 0)
        current_node_id = 0
        for pt in path:
            nmask = self.cols.parent.query("eq", current_node_id) & self.cols.name.query("__eq__", (pt,), {})
            current_node_id = cast(int, self.cols.ID.data[find_first_true(nmask)])
            if current_node_id == -1:
                return False
        return self.cols.parent.query("eq", current_node_id)


@schema
class SPos(SchemaABC):

    class required_cols(NamedTuple):
        word: PlainText

    cols: required_cols

    query_re_pat = re.compile("#(?P<start>[0-9]+)?\\.\\.(?P<stop>[0-9]+)?")

    def _query(self, mch):
        _start = mch.group("start")
        _stop = mch.group("stop")

        start = int(_start) if _start else None
        stop = int(_stop) if _stop else None

        ret = np.zeros(len(self.cols.word.data), dtype=np.bool_)
        ret[start:stop] = True

        return ret

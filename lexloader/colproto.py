import itertools
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable
from itertools import count, repeat
from typing import Any, cast

import awkward as ak
import numba
import numpy as np
from awkward.operations import str as akstr
from pinyinparser import Final, Initial, Syllable, Tone, syllables_to_str

if __package__:
    from .typing_utils import ArrayLike, AwkwardLike, asArrayLike, asAwkwardLike

    NUMBA_FILE_CACHE = True
else:
    from typing_utils import (  # type: ignore
        ArrayLike,
        AwkwardLike,
        asArrayLike,
        asAwkwardLike,
    )

    NUMBA_FILE_CACHE = False  # 防止包内调用的缓存文件把包外调用炸了。沟槽的numba为什么会有缓存依赖路径导致cwd一变缓存就炸的啥比机制


try:
    from tqdm import tqdm  # type: ignore

    tqdm = cast(Callable, tqdm)
except ImportError:

    def tqdm(_, **_____):
        return _


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ColProtoABC[T](ABC):
    aot_funcs: dict[str, Callable[[ColProtoABC], ColProtoABC]]

    def __init__(self, data: AwkwardLike) -> None:
        self.data = data

    @staticmethod
    def from_python(raw: list[T]) -> ak.Array:
        return ak.Array(raw)

    @abstractmethod
    def query(self, *args, **kwargs) -> ArrayLike: ...

    def find(self, item: T) -> ArrayLike:
        return self.data == item

    @staticmethod
    def tostr(val: T) -> str:
        return str(val)

    data: AwkwardLike


class JITColABC[T](ColProtoABC):

    def __init__(self, /, data: AwkwardLike | None = None, from_: T | None = None) -> None:
        if data is not None:
            super().__init__(data)
        elif from_ is not None:
            self.buildfrom(from_)

    @abstractmethod
    def buildfrom(self, from_: T): ...

    @staticmethod
    def from_python(*_, **_____):
        return NotImplemented


class SortedColABC(ABC):
    @staticmethod
    @numba.njit(cache=NUMBA_FILE_CACHE)
    def _bisect_bytes_l(offsets, data, target):  # sourcery skip: min-max-identity #numba用不了，下同，nputil同
        lo = 0
        hi = len(offsets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            s_start = offsets[mid]
            s_end = offsets[mid + 1]
            t_len = len(target)

            cmp = 0
            min_len = s_end - s_start
            if min_len > t_len:
                min_len = t_len

            for i in range(min_len):
                d_val = data[s_start + i]
                t_val = target[i]
                if d_val < t_val:
                    cmp = -1
                    break
                elif d_val > t_val:
                    cmp = 1
                    break

            if cmp == 0:
                if (s_end - s_start) < t_len:
                    cmp = -1
                elif (s_end - s_start) > t_len:
                    cmp = 1

            if cmp < 0:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @staticmethod
    @numba.njit(cache=NUMBA_FILE_CACHE)
    def _bisect_bytes_r(offsets, data, target):  # sourcery skip: min-max-identity
        lo = 0
        hi = len(offsets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            s_start = offsets[mid]
            s_end = offsets[mid + 1]
            t_len = len(target)

            cmp = 0
            min_len = s_end - s_start
            if min_len > t_len:
                min_len = t_len

            for i in range(min_len):
                d_val = data[s_start + i]
                t_val = target[i]
                if d_val < t_val:
                    cmp = -1
                    break
                elif d_val > t_val:
                    cmp = 1
                    break

            if cmp == 0:
                if (s_end - s_start) < t_len:
                    cmp = -1
                elif (s_end - s_start) > t_len:
                    cmp = 1

            if cmp <= 0:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @staticmethod
    def _bisect_directcmp_l(arr, t):
        lo = 0
        hi = len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @staticmethod
    def _bisect_directcmp_r(arr, t):
        lo = 0
        hi = len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] > t:
                hi = mid
            else:
                lo = mid + 1
        return lo

    __pw_a8 = np.array([0], dtype=np.int8)
    __pw_au8 = np.array([0], dtype=np.uint8)
    __pw_a16 = np.array([0], dtype=np.int16)
    __pw_au16 = np.array([0], dtype=np.uint16)
    __pw_a32 = np.array([0], dtype=np.int32)
    __pw_au32 = np.array([0], dtype=np.uint32)
    __pw_a64 = np.array([0], dtype=np.int64)
    __pw_au64 = np.array([0], dtype=np.uint64)
    __pw_af32 = np.array([0], dtype=np.float32)
    __pw_af64 = np.array([0], dtype=np.float64)
    __pw_d = np.array([], dtype=np.uint8)

    for off in (__pw_a32, __pw_a64):
        _bisect_bytes_l(off, __pw_d, __pw_au8)
        _bisect_bytes_r(off, __pw_d, __pw_au8)
    for a in tqdm(
        [__pw_a8, __pw_au8, __pw_a16, __pw_au16, __pw_a32, __pw_au32, __pw_a64, __pw_au64, __pw_af32, __pw_af64],
        desc="SortedCol NumBa JIT...",
        disable=logger.level > logging.INFO,
    ):
        _bisect_directcmp_l(a, 0)
        _bisect_directcmp_r(a, 0)


# region ColProto s


class PlainText(ColProtoABC[str]):
    def query(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> AwkwardLike:
        if method in {
            "__eq__",
            "__ne__",
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
        }:
            match method:
                case "__eq__":
                    return self.data == args[0]
                case "__ne__":
                    return self.data != args[0]
                case "__lt__":
                    return self.data < args[0]
                case "__le__":
                    return self.data <= args[0]
                case "__gt__":
                    return self.data > args[0]
                case "__ge__":
                    return self.data >= args[0]
        akstrop = getattr(akstr, method, None)
        if akstrop is None:
            raise ValueError(f"方法 {method} 不存在 @ PlainText")
        return akstrop(self.data, *args, **kwargs)

    def aot_Pinyin(self) -> Pinyin:  # 能jit的一定能aot，但是能aot的未必适合jit（比如拼音这种东西只适合aot）
        import pinyinparser
        import pypinyin
        from pypinyin_dict.phrase_pinyin_data import cc_cedict, large_pinyin

        cc_cedict.load()
        large_pinyin.load()

        return Pinyin(
            cast(
                AwkwardLike,
                Pinyin.from_python(
                    [
                        [
                            (0 if s is None else int(pinyinparser.parse_single(s)))
                            for s in pypinyin.lazy_pinyin(
                                cast(str, word),
                                style=pypinyin.Style.TONE,
                                errors=lambda _: None,  # type: ignore #沟槽的pypinyin类型注解少写了怎么到现在还没改
                            )
                        ]
                        for word in tqdm(self.data)
                    ]
                ),
            )
        )


class PlainTextS(PlainText, SortedColABC):  # PT的有序版本，下同。主要是多了个二分，别的都直接继承了
    def query(self, method: str, args: tuple[Any], kwargs: dict[str, Any]) -> AwkwardLike:
        if method not in {
            "__eq__",
            "__ne__",
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
        }:
            return super().query(method, args, kwargs)
        target_bytes = np.frombuffer(args[0].encode("utf8"), dtype=np.uint8)
        offsets = self.data.layout.offsets.data
        content = self.data.layout.content.data
        ret = np.empty(len(self.data), dtype=np.bool_)
        match method:
            case "__eq__":
                ll = self._bisect_bytes_l(offsets, content, target_bytes)
                rr = self._bisect_bytes_r(offsets, content, target_bytes)
                ret[:ll] = False
                ret[ll:rr] = True
                ret[rr:] = False
            case "__ne__":
                ll = self._bisect_bytes_l(offsets, content, target_bytes)
                rr = self._bisect_bytes_r(offsets, content, target_bytes)
                ret[:ll] = True
                ret[ll:rr] = False
                ret[rr:] = True
            case "__lt__":
                ll = self._bisect_bytes_l(offsets, content, target_bytes)
                ret[:ll] = True
                ret[ll:] = False
            case "__le__":
                rr = self._bisect_bytes_r(offsets, content, target_bytes)
                ret[:rr] = True
                ret[rr:] = False
            case "__gt__":
                rr = self._bisect_bytes_r(offsets, content, target_bytes)
                ret[:rr] = False
                ret[rr:] = True
            case "__ge__":
                ll = self._bisect_bytes_l(offsets, content, target_bytes)
                ret[:ll] = False
                ret[ll:] = True
        return asAwkwardLike(ak.Array(ret))


class _Int(ColProtoABC[int]):
    def query(self, method: str, target: int) -> AwkwardLike:
        match method:
            case "eq":
                return self.data == target
            case "ne":
                return self.data != target
            case "gt":
                return self.data > target
            case "ge":
                return self.data >= target
            case "lt":
                return self.data < target
            case "le":
                return self.data <= target
        raise ValueError(f"方法 {method} 不存在 @ Int")


class _IntS(_Int, SortedColABC):
    def query(self, method: str, target: int) -> AwkwardLike:
        # sourcery skip: extract-duplicate-method
        ret = np.empty(len(self.data), dtype=np.bool_)
        match method:
            case "eq":
                ll = self._bisect_directcmp_l(self.data, target)
                rr = self._bisect_directcmp_r(self.data, target)
                ret[:ll] = False
                ret[ll:rr] = True
                ret[rr:] = False
            case "ne":
                ll = self._bisect_directcmp_l(self.data, target)
                rr = self._bisect_directcmp_r(self.data, target)
                ret[:ll] = True
                ret[ll:rr] = False
                ret[rr:] = True
            case "lt":
                ll = self._bisect_directcmp_l(self.data, target)
                ret[:ll] = True
                ret[ll:] = False
            case "le":
                rr = self._bisect_directcmp_r(self.data, target)
                ret[:rr] = True
                ret[rr:] = False
            case "gt":
                rr = self._bisect_directcmp_r(self.data, target)
                ret[:rr] = False
                ret[rr:] = True
            case "ge":
                ll = self._bisect_directcmp_l(self.data, target)
                ret[:ll] = False
                ret[ll:] = True
            case _:
                raise ValueError(f"方法 {method} 不存在 @ IntS")
        return asAwkwardLike(ret)


class Int64(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.int64))


class UInt64(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.uint64))


class Int32(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.int32))


class UInt32(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.uint32))


class Int16(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.int16))


class UInt16(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.uint16))


class Int8(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.int8))


class UInt8(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.uint8))


class Int64S(Int64, _IntS):
    pass


class UInt64S(UInt64, _IntS):
    pass


class Int32S(Int32, _IntS):
    pass


class UInt32S(UInt32, _IntS):
    pass


class Int16S(Int16, _IntS):
    pass


class UInt16S(UInt16, _IntS):
    pass


class Int8S(Int8, _IntS):
    pass


class UInt8S(UInt8, _IntS):
    pass


class Bool(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.bool_))


class _Float(ColProtoABC[float]):
    def query(self, method: str, target: float) -> AwkwardLike:
        match method:
            case "eq":
                return self.data == target
            case "ne":
                return self.data != target
            case "gt":
                return self.data > target
            case "ge":
                return self.data >= target
            case "lt":
                return self.data < target
            case "le":
                return self.data <= target
        raise ValueError(f"方法 {method} 不存在 @ Float")


class _FloatS(_Float, SortedColABC):  # 其实和_IntS没有任何区别xwx懒得合并了，之后啥时候想起来再来合并
    def query(self, method: str, target: float) -> AwkwardLike:
        # sourcery skip: extract-duplicate-method
        ret = np.empty(len(self.data), dtype=np.bool_)
        match method:
            case "eq":
                ll = self._bisect_directcmp_l(self.data, target)
                rr = self._bisect_directcmp_r(self.data, target)
                ret[:ll] = False
                ret[ll:rr] = True
                ret[rr:] = False
            case "ne":
                ll = self._bisect_directcmp_l(self.data, target)
                rr = self._bisect_directcmp_r(self.data, target)
                ret[:ll] = True
                ret[ll:rr] = False
                ret[rr:] = True
            case "lt":
                ll = self._bisect_directcmp_l(self.data, target)
                ret[:ll] = True
                ret[ll:] = False
            case "le":
                rr = self._bisect_directcmp_r(self.data, target)
                ret[:rr] = True
                ret[rr:] = False
            case "gt":
                rr = self._bisect_directcmp_r(self.data, target)
                ret[:rr] = False
                ret[rr:] = True
            case "ge":
                ll = self._bisect_directcmp_l(self.data, target)
                ret[:ll] = False
                ret[ll:] = True
            case _:
                raise ValueError(f"方法 {method} 不存在 @ FloatS")
        return asAwkwardLike(ret)


class Float32(_Float):
    @staticmethod
    def from_python(raw: list[float]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.float32))


class Float64(_Float):
    @staticmethod
    def from_python(raw: list[float]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.float64))


class _Complex(ColProtoABC[complex]):
    @staticmethod
    def tostr(val: complex) -> str:
        return f"{val.real}+{val.imag}i"

    def query(self, method: str, target: complex) -> AwkwardLike:
        match method:
            case "eq":
                return self.data == target
            case "ne":
                return self.data != target
        raise ValueError(f"方法 {method} 不存在 @ Complex")


class Complex64(_Complex):
    @staticmethod
    def from_python(raw: list[complex]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.complex64))


class Complex128(_Complex):
    @staticmethod
    def from_python(raw: list[complex]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.complex128))


class Pinyin(ColProtoABC[list[int]]):
    @staticmethod
    def tostr(val: list[int]) -> str:
        return syllables_to_str((Syllable(int(v)) for v in val), "'")

    @staticmethod
    def from_python(raw: list[list[int]]) -> ak.Array:
        data = asAwkwardLike(ak.values_astype(ak.Array(raw), "uint16"))
        offsets = np.asarray(data.layout.offsets)
        max_offset = offsets[-1]
        i32max = np.iinfo(np.int32).max
        if max_offset <= i32max:
            new_offsets = offsets.astype(np.int32)
        else:
            new_offsets = offsets.astype(np.int64)
        new_index = ak.index.Index(new_offsets)
        new_layout = ak.contents.ListOffsetArray(new_index, data.layout.content)
        return ak.Array(new_layout)

    @staticmethod
    @numba.njit(parallel=True, cache=NUMBA_FILE_CACHE)
    def __p_check(offsets, data16, m: int | None, n: int | None, and16, xand16):
        nstring = len(offsets) - 1
        result = np.zeros(nstring, dtype=np.bool_)
        _m = 0 if m is None else m
        if _m < 0:
            if n is None:
                # 后缀 不定长 [-1:]syl
                q = -_m
                if len(and16) != q:
                    raise ValueError
                for i in numba.prange(nstring):
                    if (offsets[i + 1] - offsets[i]) < q:
                        continue
                    base = offsets[i + 1] + _m
                    for j in range(q):
                        if (data16[base + j] & and16[j]) != xand16[j]:
                            break
                    else:
                        result[i] = True
            elif n < 0:
                # 中缀 尾定位 不定长[-2:-1]syl
                q = n - _m
                if len(and16) != q:
                    raise ValueError
                l = -_m
                for i in numba.prange(nstring):
                    if (offsets[i + 1] - offsets[i]) < l:
                        continue
                    base = offsets[i + 1] + _m
                    for j in range(q):
                        if (data16[base + j] & and16[j]) != xand16[j]:
                            break
                    else:
                        result[i] = True
            else:
                raise IndexError
        elif n is None:
            # 全匹配 [:]syl / 后缀 定长 [1:]syl
            q = len(and16)
            l = _m + q
            for i in numba.prange(nstring):
                if (offsets[i + 1] - offsets[i]) != l:
                    continue
                base = offsets[i] + _m
                for j in range(q):
                    if (data16[base + j] & and16[j]) != xand16[j]:
                        break
                else:
                    result[i] = True
        elif n >= 0:
            # 前缀 不定长 [:1]syl / 中缀 头定位 不定长 [1:2]syl
            q = n - _m
            if len(and16) != q:
                raise ValueError
            for i in numba.prange(nstring):
                if (offsets[i + 1] - offsets[i]) < n:
                    continue
                base = offsets[i] + _m
                for j in range(q):
                    if (data16[base + j] & and16[j]) != xand16[j]:
                        break
                else:
                    result[i] = True
        else:
            # 前缀 定长 [:-1]syl / 中缀 定位 定长 [1:-1]syl
            q = len(and16)
            l = _m + q - n
            for i in numba.prange(nstring):
                if (offsets[i + 1] - offsets[i]) != l:
                    continue
                base = offsets[i] + _m
                for j in range(q):
                    if (data16[base + j] & and16[j]) != xand16[j]:
                        break
                else:
                    result[i] = True
        return result

    __pw_o64 = np.array([0], dtype=np.int64)
    __pw_o32 = np.array([0], dtype=np.int32)
    __pw_d = np.array([], dtype=np.uint16)
    __pw_a = np.array([0], dtype=np.uint16)
    __pw_x = np.array([0], dtype=np.uint16)
    for __c_o, (__c_m, __c_n) in tqdm(
        itertools.product([__pw_o32, __pw_o64], [(0, None), (0, -1), (0, 1), (None, None), (None, -1), (None, 1), (-1, None), (-2, -1)]),
        total=16,
        desc="PinYin NumBa JIT...",
        disable=logger.level > logging.INFO,
    ):
        __p_check(__c_o, __pw_d, __c_m, __c_n, __pw_a, __pw_x)

    def query(self, m: int | None, n: int | None, iw: Iterable[bool] | bool, fw: Iterable[bool] | bool, s: Iterable[Syllable]) -> ArrayLike:
        print(f"m={repr(m)}, n={repr(n)}, iw={repr(iw)}, fw={repr(fw)}, s={repr(s)}")
        offset: ArrayLike = self.data.layout.offsets.data
        data: ArrayLike = self.data.layout.content.data

        and_lst: list[np.uint16] = []
        xor_lst: list[np.uint16] = []

        iw = repeat(iw) if isinstance(iw, bool) else iw
        fw = repeat(fw) if isinstance(fw, bool) else fw

        for syll, iwp, fwp in zip(s, iw, fw):
            and_val = 0
            xor_val = int(syll)

            if syll.initial not in (Initial.missing, Initial.unspec):
                and_val |= 0x001F if iwp else 0x801F
            if syll.final not in (Final.missing, Final.unspec):
                and_val |= 0x1F00 if fwp else 0x7F00
            if syll.tone not in (Tone.missing, Tone.unspec):
                and_val |= 0x00E0

            and_lst.append(np.uint16(and_val))
            xor_lst.append(np.uint16(xor_val))

        print(f"{and_lst=} {xor_lst=}")

        and_np = np.array(and_lst, dtype=np.uint16)
        xand_np = np.array(xor_lst, dtype=np.uint16) & and_np

        return asArrayLike(self.__p_check(offset, data, m, n, and_np, xand_np))


class _Enum(_Int):
    @staticmethod
    def _from_python(raw: list[str], dtype) -> ak.Array:
        enummap: defaultdict[str, int] = defaultdict(count(1).__next__)
        data = np.empty(len(raw), dtype=dtype)
        for i, s in enumerate(raw):
            data[i] = enummap[s]
        return ak.with_parameter(ak.Array(data), "enummap", enummap)

    @staticmethod
    @abstractmethod
    def from_python(raw: list[str]) -> ak.Array:
        pass

    def query(self, method: str, target: str):
        enummap: dict[str, int] = cast(dict[str, int], ak.parameters(self.data)["enummap"])
        match method:
            case "eq":
                if target not in enummap:
                    return ak.Array(np.zeros(len(self.data), dtype=np.bool_))
                return self.data == enummap[target]
            case "ne":
                if target not in enummap:
                    return ak.Array(np.ones(len(self.data), dtype=np.bool_))
                return self.data != enummap[target]
            case _:
                raise ValueError(f"方法 {method} 不存在 @ Enum")


class Enum8(_Enum):
    @staticmethod
    def from_python(raw: list[str]) -> ak.Array:
        return _Enum._from_python(raw, np.uint8)


class Enum16(_Enum):
    @staticmethod
    def from_python(raw: list[str]) -> ak.Array:
        return _Enum._from_python(raw, np.uint16)


class Enum32(_Enum):
    @staticmethod
    def from_python(raw: list[str]) -> ak.Array:
        return _Enum._from_python(raw, np.uint32)


class Enum64(_Enum):
    @staticmethod
    def from_python(raw: list[str]) -> ak.Array:
        return _Enum._from_python(raw, np.uint64)


class _Length(JITColABC[PlainText], _Int):
    @staticmethod
    @abstractmethod
    def _p_len(offset, data): ...

    def buildfrom(self, from_: PlainText) -> None:
        self.data = asAwkwardLike(ak.Array(self._p_len(from_.data.layout.offsets.data, from_.data.layout.content.data)))


class Length8(_Length, UInt8):
    @staticmethod
    @numba.njit(parallel=True, cache=NUMBA_FILE_CACHE)
    def _p_len(offsets, data):
        nstr = len(offsets) - 1
        lengths = np.empty(nstr, dtype=np.uint8)
        for i in numba.prange(nstr):
            cnt = np.uint8(0)
            for j in range(offsets[i], offsets[i + 1]):
                if (data[j] & 0xC0) != 0x80:
                    cnt += 1
            lengths[i] = cnt
        return lengths

    _p_len(np.array([0], dtype=np.int64), np.array([], dtype=np.uint8))


class Length16(_Length, UInt16):
    @staticmethod
    @numba.njit(parallel=True, cache=NUMBA_FILE_CACHE)
    def _p_len(offsets, data):  # DRY不了，numba没法推断这种外源类型，所以只能整个copy一遍，唉太坏了
        nstr = len(offsets) - 1
        lengths = np.empty(nstr, dtype=np.uint16)
        for i in numba.prange(nstr):
            cnt = np.uint16(0)
            for j in range(offsets[i], offsets[i + 1]):
                if (data[j] & 0xC0) != 0x80:
                    cnt += 1
            lengths[i] = cnt
        return lengths

    _p_len(np.array([0], dtype=np.int64), np.array([], dtype=np.uint8))


class Length32(_Length, UInt32):
    @staticmethod
    @numba.njit(parallel=True, cache=NUMBA_FILE_CACHE)
    def _p_len(offsets, data):
        nstr = len(offsets) - 1
        lengths = np.empty(nstr, dtype=np.uint32)
        for i in numba.prange(nstr):
            cnt = np.uint32(0)
            for j in range(offsets[i], offsets[i + 1]):
                if (data[j] & 0xC0) != 0x80:
                    cnt += 1
            lengths[i] = cnt
        return lengths

    _p_len(np.array([0], dtype=np.int64), np.array([], dtype=np.uint8))


class Length64(_Length, UInt64):
    @staticmethod
    @numba.njit(parallel=True, cache=NUMBA_FILE_CACHE)
    def _p_len(offsets, data):
        nstr = len(offsets) - 1
        lengths = np.empty(nstr, dtype=np.uint64)
        for i in numba.prange(nstr):
            cnt = np.uint64(0)
            for j in range(offsets[i], offsets[i + 1]):
                if (data[j] & 0xC0) != 0x80:
                    cnt += 1
            lengths[i] = cnt
        return lengths

    _p_len(np.array([0], dtype=np.int64), np.array([], dtype=np.uint8))


PROTO_NAMES: dict[str, type[ColProtoABC]] = {
    "PT": PlainText,
    "PTS": PlainTextS,
    "B": Bool,
    "U8": UInt8,
    "I8": Int8,
    "U16": UInt16,
    "I16": Int16,
    "U32": UInt32,
    "I32": Int32,
    "U64": UInt64,
    "I64": Int64,
    "U8S": UInt8S,
    "I8S": Int8S,
    "U16S": UInt16S,
    "I16S": Int16S,
    "U32S": UInt32S,
    "I32S": Int32S,
    "U64S": UInt64S,
    "I64S": Int64S,
    "F32": Float32,
    "F64": Float64,
    "C64": Complex64,
    "C128": Complex128,
    "EN8": Enum8,
    "EN16": Enum16,
    "EN32": Enum32,
    "EN64": Enum64,
    "PY": Pinyin,
}

JIT_NAMES: dict[str, type[JITColABC]] = {
    "L8": Length8,
    "L16": Length16,
    "L32": Length32,
    "L64": Length64,
}

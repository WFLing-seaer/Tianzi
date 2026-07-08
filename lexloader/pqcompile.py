import csv
import re
import sys
import typing
from collections.abc import Callable, Iterable
from itertools import chain
from pathlib import Path

import awkward as ak
import numpy as np
from pinyinparser import parse as pinyin_parse
from tqdm import tqdm
from typing_utils import AwkwardLike

try:
    from .colproto import (
        JIT_NAMES,
        PROTO_NAMES,
        Bool,
        ColProtoABC,
        Pinyin,
        SortedColABC,
        _Enum,
    )
    from .headparser import compile_get_parts, compile_parse
except ModuleNotFoundError, ImportError:
    from colproto import (
        JIT_NAMES,
        PROTO_NAMES,
        Bool,
        ColProtoABC,
        Pinyin,
        SortedColABC,
        _Enum,
    )
    from headparser import compile_get_parts, compile_parse


def get_type(cls: type) -> type | None:
    for klass in cls.__mro__:
        if orig_bases := getattr(klass, "__orig_bases__", None):
            for ob in orig_bases:
                if args := typing.get_args(ob):
                    return args[0]
    return None


def _cvrt_Pinyin(raw_values: list[str]) -> ak.Array:
    parsed: list[list[int]] = []
    for val in tqdm(raw_values):
        if val.strip():
            syls = pinyin_parse(val, default_tone_neutral=True, force_valid_syllable=True, missing_as_nul=True)
            parsed.append([int(s) for s in syls])
        else:
            parsed.append([])
    return Pinyin.from_python(parsed)


def _cvrt_Bool(raw_values: list[str]) -> ak.Array:
    parsed: list[int] = []
    for val in raw_values:
        val_lower = val.strip().lower()
        if val_lower in ("true", "1", "t", "yes", "y"):
            parsed.append(1)
        elif val_lower in ("false", "0", "f", "no", "n", ""):
            parsed.append(0)
        else:
            raise ValueError(f"'{val}' -> Bool 解析不能")
    return Bool.from_python(parsed)


NONTRIVAL_CVRT: dict[type[ColProtoABC], Callable[..., ak.Array]] = {
    Pinyin: _cvrt_Pinyin,
    Bool: _cvrt_Bool,
}


def convert_raw(proto_cls: type[ColProtoABC], raw_values: list[str]) -> ak.Array:
    if proto_cls in NONTRIVAL_CVRT:
        return NONTRIVAL_CVRT[proto_cls](raw_values)
    if issubclass(proto_cls, _Enum):
        return proto_cls.from_python(raw_values)
    type_param = get_type(proto_cls)
    if type_param is None:
        raise ValueError(f"推断{proto_cls.__name__}的类型失败，如果这个是你自己加的ColProto建议在上边nontrivial里把这个加进去")
    if type_param is str:
        return proto_cls.from_python(raw_values)
    try:
        parsed = [type_param(v) if v else type_param() for v in raw_values]
        return proto_cls.from_python(parsed)
    except TypeError as e:
        raise ValueError(f"{type_param} -> {proto_cls} 解析不能") from e


def compile_dsv(lines: Iterable[str], delim: str = ",") -> ak.Array:
    # sourcery skip: extract-method
    reader = csv.DictReader(lines, delimiter=delim)
    headers = reader.fieldnames
    if not headers:
        raise ValueError("词库源文件必须有表头")
    raw_data: dict[str, list[str]] = {h: [] for h in headers}
    for row in reader:
        for h in headers:
            raw_data[h].append(row.get(h, ""))

    result_arrs: dict[str, AwkwardLike] = {}
    scol0_name: str | None = None
    sorted_keys: list[str] = []

    for header, rawheader in chain(
        ((header[1:], header) for header in headers if header.startswith("*")),
        ((header, header) for header in headers if not header.startswith("*")),
    ):
        compiling_heads = compile_parse(header)
        main_str, aot_strs = compile_get_parts(header)
        main_proto_cls = PROTO_NAMES.get(compiling_heads.main.proto.upper())
        if main_proto_cls is None:
            raise ValueError(f"未知的协议类型: {compiling_heads.main.proto} @ col:'{rawheader}'")

        if issubclass(main_proto_cls, SortedColABC):
            sorted_keys.append(main_str)
            if scol0_name is None:
                scol0_name = main_str

        main_arr = convert_raw(main_proto_cls, raw_data[rawheader])
        result_arrs[main_str] = typing.cast(AwkwardLike, main_arr)

        if not compiling_heads.aot:
            continue

        main_col = main_proto_cls(typing.cast(AwkwardLike, main_arr))
        group_match = re.match(r"\{.+?\}", main_str)
        main_group = group_match[0] if group_match else None
        for aot_idt, aot_str in zip(compiling_heads.aot, aot_strs):
            aot_proto = aot_idt.proto.upper()
            aot_col: ColProtoABC | None = None
            aot_proto_cls = PROTO_NAMES.get(aot_proto)
            if aot_proto_cls is not None:
                aot_func = getattr(main_proto_cls, f"aot_{aot_proto_cls.__name__}", None)
                if aot_func is not None:
                    aot_col = aot_func(main_col)
            if aot_col is None:
                jit_cls = JIT_NAMES.get(aot_proto)
                if jit_cls is not None:
                    aot_col = jit_cls(from_=main_col)
            if aot_col is None:
                if aot_proto_cls is not None:
                    raise ValueError(f"无法从 {main_proto_cls.__name__} AOT 编译到 {aot_proto_cls.__name__} @ aot:'{aot_str}'")
                else:
                    raise ValueError(f"未知的协议类型: {aot_idt.proto} @ aot:'{aot_str}'")

            if not aot_str.startswith("{") and main_group:
                aot_str = main_group + aot_str
            result_arrs[aot_str] = aot_col.data

            if aot_proto_cls is not None and issubclass(aot_proto_cls, SortedColABC):
                sorted_keys.append(aot_str)

    if scol0_name is not None:
        scol0 = result_arrs[scol0_name]
        sort_idx = ak.argsort(scol0)
        n = len(scol0)

        scol0_already_sorted = n <= 1 or np.array_equal(ak.to_numpy(sort_idx), np.arange(n))

        if not scol0_already_sorted:
            for key in list(result_arrs.keys()):
                result_arrs[key] = result_arrs[key][sort_idx]

        for key in sorted_keys:
            if key == scol0_name:
                continue
            arr = result_arrs[key]
            n_other = len(arr)
            if n_other > 1:
                other_sort_idx = ak.argsort(arr)
                if not np.array_equal(ak.to_numpy(other_sort_idx), np.arange(n_other)):
                    raise ValueError(f"列 '{scol0_name}' 与 '{key}' 无法同时有序")

    return ak.Array(result_arrs)


if __name__ == "__main__" and (len(sys.argv) > 1 and sys.argv[1]):
    fn = Path(sys.argv[1])
    delim = {".csv": ",", ".tsv": "\t"}[fn.suffix]
    with open(fn, "r", encoding="utf-8") as f:
        arr = compile_dsv(f, delim)
    ak.to_parquet(arr, fn.with_suffix(".pq"))

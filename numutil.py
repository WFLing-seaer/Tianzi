import contextlib
from cmath import isinf, isnan
from collections.abc import Iterable
from enum import IntEnum, StrEnum
from typing import Literal
from unicodedata import normalize

import cn2an
import rn2an


class NILVL(IntEnum):
    N = 0
    X = 1
    F = 2
    I = 3
    C = 4
    R = 5
    U = 6


class NSLVL(StrEnum):
    C = "c"
    U = "u"
    N = "n"
    R = "r"
    X = "x"
    XU = "X"


Value = int | float | complex
type Real = int | float
type Numeral = tuple[Literal["d", "x", "X", "f", "", "s"], Literal["c", "C", "u", "n", "r", "R"], Value, NILVL]  # ftype output_mode value NILVL
type Numerals = tuple[Literal["d", "x", "X", "f", "", "s"], Literal["c", "C", "u", "n", "r", "R"], list[Value], int]


def numify(text: str, suppress_overflow: bool = False, lvl: NILVL | NSLVL | None = None, sugar: bool = True) -> Numeral:
    text = normalize("NFKC", text)
    if sugar and (lvl is None):
        lvl = {"0x": NILVL.X, "0f": NILVL.F, "0i": NILVL.I, "0r": NILVL.R, "0u": NILVL.U}.get(text[:2])
        # 语法糖。复数为0i以避免和十六进制0c混淆。注意开启语法糖会导致0xXXX的类型由N变为X。0bXXX、0oXXX不受此影响。
        if lvl is not None:
            text = text[2:]
    with contextlib.suppress(ValueError):
        if (lvl is None) or (lvl == NILVL.N) or (lvl == NSLVL.N):
            return "d", "n", int(text), NILVL.N
    with contextlib.suppress(ValueError):
        if ((lvl is None) or (lvl == NILVL.X) or (lvl == NSLVL.X)) and (text.islower() or text.isdecimal()):
            return "x", "n", int(text, 16), NILVL.X
    with contextlib.suppress(ValueError):
        if (lvl is None) or (lvl == NILVL.F) or (lvl == NSLVL.N):
            val = float(text)
            if (not suppress_overflow) and (isinf(val) or isnan(val)):
                raise OverflowError
            return "f", "n", val, NILVL.F
    with contextlib.suppress(ValueError):
        if (lvl is None) or (lvl == NILVL.I) or (lvl == NSLVL.N):
            val = complex(text)
            if (not suppress_overflow) and (isinf(val.real) or isnan(val.real) or isinf(val.imag) or isnan(val.imag)):
                raise OverflowError
            return "", "n", val, NILVL.I
    with contextlib.suppress(ValueError):
        if (lvl is None) or (lvl == NILVL.C) or (lvl == NSLVL.C):
            with contextlib.suppress(ValueError):
                return "d", "c", int(text), NILVL.C  # 因为cn2an会把原本就是阿拉伯数字的整数转成浮点数，因此此处通过int将这种情况短路掉
            x = cn2an.cn2an(text, "smart")
            try:
                return "d", "c", int(x), NILVL.C
            except ValueError:
                return "f", "c", float(x), NILVL.C
    with contextlib.suppress(ValueError):
        if (lvl is None) or (lvl == NILVL.R) or (lvl == NSLVL.R):
            return "f", "r", rn2an.rn2an(text), NILVL.R
    with contextlib.suppress(ValueError):
        if (lvl is None) or (lvl == NILVL.X) or (lvl == NSLVL.XU):
            return "X", "n", int(text, 16), NILVL.X
    with contextlib.suppress(TypeError):
        if (lvl is None) or (lvl == NILVL.U) or (lvl == NSLVL.U):
            return "s", "u", ord(text), NILVL.U
    raise ValueError


def as_numeral(n: Value, _minlvl: NILVL | None = None) -> Numeral:
    if isinstance(n, int) or (_minlvl is not None and _minlvl < 2):
        return "d", "n", int(n.real), _minlvl or NILVL.N
    if isinstance(n, float) or _minlvl in [2, 4, 5]:
        return "f", "n", float(n.real), _minlvl or NILVL.N
    if isinstance(n, complex) or _minlvl == 3:
        return "", "n", n, _minlvl or NILVL.N
    if _minlvl and _minlvl == 6:
        return "s", "u", int(n), 6


def numsify(texts: Iterable[str], suppress_overflow: bool = False) -> Numerals:
    texts_lst = list(texts)
    default_parse = [numify(i, suppress_overflow) for i in texts_lst]
    lvls = {num[3] for num in default_parse}

    if len(lvls) == 1:
        return (default_parse[0][0], default_parse[0][1], [i[2] for i in default_parse], lvls.pop())

    for trial in sorted(lvls):
        trial_result = []
        for idx, text in enumerate(texts_lst):
            if default_parse[idx][3] == trial:
                trial_result.append(default_parse[idx])
            else:
                with contextlib.suppress(ValueError):
                    trial_result.append(numify(text, suppress_overflow, trial))
                    continue
                break
        else:
            return (trial_result[0][0], trial_result[0][1], [i[2] for i in trial_result], trial)

    lvl = max(lvls)
    nums = [numify(i, suppress_overflow, lvl) for i in texts_lst]
    return (nums[0][0], nums[0][1], [i[2] for i in nums], lvl)


def numsimp(n: Value) -> Value:
    if isinstance(n, complex):
        n = n if n.imag else n.real
    if isinstance(n, float):
        n = int(n) if not isnan(n) and not isinf(n) and n == int(n) else n
    return n


def numfmt(
    output_mode: Literal["c", "C", "u", "n", "r", "R"],
    fmt_spec: str,
    fill: str,
    align: str,
    sign: str,
    perc: int | None,
    ftype: str,
    value: Value,
) -> str:
    match output_mode:
        case "c" | "C":
            if isinstance(value, complex):
                raise ValueError
            # 执行对中文数字的格式化。这个好像没有现成的库可以用，只能自己写
            if ftype == "%":
                value *= 100
            if perc is not None:
                value = round(value, perc)
            an2cn_format = "direct" if ftype == "s" else {"c": "low", "C": "up"}[output_mode]
            ret = cn2an.an2cn(str(value), an2cn_format)
            if sign == " ":
                ret = "\u3000" + ret if ret[0] != "负" else ret
            elif sign == "+":
                ret = f"正{ret}" if ret[0] != "负" else ret
            ret = f"{{r:{fill}{align}}}".format(r=ret)
            if perc:
                if "点" not in ret:
                    ret += "点"
                ret += "零" * (perc - len(ret.split("点")[-1]))
            if ftype == "%":
                ret = f"佰分之{ret}" if an2cn_format == "up" else f"百分之{ret}"
            return ret
        case "u":
            if not isinstance(value, int):
                raise ValueError
            if not (0 <= value <= 0x10FFFF):
                raise UnicodeError
            ret = chr(value)
        case "r":
            if isinstance(value, complex):
                raise ValueError
            ret = rn2an.an2rnA(value)
        case "R":
            if isinstance(value, complex):
                raise ValueError
            ret = rn2an.an2rn(value)
        case _:
            ret = str(value)
    if fmt_spec:
        ret = f"{{v:{fmt_spec}}}".format(v=value)
    return ret

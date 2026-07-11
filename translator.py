import ast
import asyncio
import contextlib
import numbers
import secrets
import time
import traceback
import warnings
import weakref
from collections import deque
from collections.abc import Callable, Coroutine, Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from itertools import pairwise, repeat
from logging import getLogger
from random import Random
from types import NoneType, SimpleNamespace
from typing import (
    Any,
    AnyStr,
    Literal,
    Never,
    Protocol,
    SupportsIndex,
    TypeAlias,
    cast,
    no_type_check,
    runtime_checkable,
)

import ANSWER_TO_THE_ULTIMATE_QUESTION_OF_LIFE_THE_UNIVERSE_AND_EVERYTHING
import awkward
import numpy as np
import regex
import simpleeval
from cachetools import LRUCache, TTLCache
from regex import Match as _Match
from regex import Pattern as _Pattern
from regex import escape

from . import lexloader
from .cacheutil import Cache
from .fontutil import fonts
from .numutil import NSLVL, Value, numfmt, numify, numsify, numsimp
from .pysandbox import RETVAL_MISSING, Sandbox
from .randutil import rngs, rsgs
from .textutil import find_outmost_bracket

# region global settings

simpleeval.MAX_POWER = 1024
simpleeval.MAX_COMPREHENSION_LENGTH = 1024
simpleeval.MAX_STRING_LENGTH = 4096
simpleeval.MAX_SHIFT = 1024

# endregion

# region symbols

SYM_HEAD = "[["
SYM_TAIL = "]]"
SYM_MODIFY = "@"
SYM_LENGTH = "="
SYM_CACHE = ">"

CS_RANGE = "-~－～—到至"  # CharSet
CS_SEGSEP = ";；|/"
CS_SPLITSEP = "，、 "
CS_SEP = "，、；？,.; \n"
CS_INLINE = "\ue104-\ue500"

SYN_LA = ">>>"  # SYNtax LingerAssign
SYN_LR = "<<<"
SYN_SLCSEP = ":"
SYN_CALC = "="
SYN_IVA = ">>"
SYN_IVR = "<<"
SYN_IVFIELD = ":"
SYN_REF = "<"  # 没有SYN_ASSIGN，用的是SYM_CACHE
SYN_FONT = ":"
SYN_IN = "#"
SYN_REPEAT = "*"
SYN_REPEATL = "**"

SYN_STRICTMODE = "$$"


RSYM_HEAD = escape(SYM_HEAD)
RSYM_TAIL = escape(SYM_TAIL)
RSYM_MODIFY = escape(SYM_MODIFY)
RSYM_LENGTH = escape(SYM_LENGTH)
RSYM_CACHE = escape(SYM_CACHE)
RCS_RANGE = escape(CS_RANGE)
RCS_SEGSEP = escape(CS_SEGSEP)
RCS_SPLITSEP = escape(CS_SPLITSEP)
RCS_INLINE = escape(CS_INLINE)
RSYN_LA = escape(SYN_LA)
RSYN_LR = escape(SYN_LR)
RSYN_SLICESEP = escape(SYN_SLCSEP)
RSYN_CALC = escape(SYN_CALC)
RSYN_IVA = escape(SYN_IVA)
RSYN_IVR = escape(SYN_IVR)
RSYN_IVFIELD = escape(SYN_IVFIELD)
RSYN_REF = escape(SYN_REF)
RSYN_FONT = escape(SYN_FONT)
RSYN_IN = escape(SYN_IN)
RSYN_REPEAT = escape(SYN_REPEAT)
RSYN_REPEATL = escape(SYN_REPEATL)

RCS_SEP = "\\s\\，\\、\\；\\？\\,\\;"  # 此处不用escape是因为\s会被escape转义失效
RCG_ELLIPSIS = "\\.\\.\\.|\\.\\.\\.\\.\\.\\.|\\…|\\…\\…|\\。\\。\\。|\\-|\\~|\\－|\\～|\\—|\\*"
RCS_CJK = (
    "\u3007\u4e00-\u9fff\ufa0e\ufa0f\ufa11\ufa13\ufa14\ufa1f\ufa21\ufa23\ufa24\ufa27-\ufa29\u3300-\u4dbf"
    "\U00020000-\U0002a6df\U0002a700-\U0002b739\U0002b740-\U0002b81d\U0002b820-\U0002cea1\U0002ceb0-\U0002ebe0\U00030000-\U0003134a\U00031350-\U000323af\U0002ebf0-\U0002ee5d\U000323b0-\U0003347f"
)

# endregion

# region global defs

type Match = _Match[str]
type Pattern = _Pattern[str]
type Translator = Callable[[Tianzi, SupportsGroup], Coroutine[None, None, SupportsStr]]
MaybeNone: TypeAlias = Any  # typeshed就是这么写的……我直接cv了


class BreakOut(Exception):
    pass


class PosteriorReject(Exception):
    pass


class WhatTheFuckIsThis(Exception):
    pass


class PUACharDrained(Exception):
    pass


@runtime_checkable
class SupportsStr(Protocol):
    def __str__(self) -> str: ...


@dataclass
class _Stat:
    err_level: Literal["ignore", "abort", "inline", "raise"] = "raise"
    # ignore: 替换成"" abort: 保持原样 inline: 行内替换为缩写 raise: 抛出

    allow_underscore_in_cache_name: bool = False

    max_calc_output_length = 512

    allow_calc_big_number: bool = False

    censor: bool = True

    allow_pua_warning = True
    pua_warning: bool = False


@runtime_checkable
class SupportsGroup(Protocol):
    def group(self, group: str | int, /) -> AnyStr | MaybeNone: ...

    def groupdict(self) -> dict[str, str | None]: ...


class DuckMatch:
    def __init__(self, gd: dict[str, str | None]):
        self._gd = gd

    def group(self, group: SupportsIndex | str = 0) -> str | None:
        if group == 0:
            return "".join(v for v in self._gd.values() if v is not None)
        return self._gd.get(group) if isinstance(group, str) else None

    def groupdict(self) -> dict[str, str | None]:
        return self._gd | {"__mock__": repr(self)}


# endregion

# region global consts

random = Random(ANSWER_TO_THE_ULTIMATE_QUESTION_OF_LIFE_THE_UNIVERSE_AND_EVERYTHING.ANSWER)
nrandom = np.random.default_rng(ANSWER_TO_THE_ULTIMATE_QUESTION_OF_LIFE_THE_UNIVERSE_AND_EVERYTHING.ANSWER)

logger = getLogger("translators")

EPACSE = {n: n - 0xE000 for n in range(0xE000, 0xE080)}

# endregion

# region global vars

translators: list[tuple[Pattern, Translator]] = []

LINGERS: TTLCache = TTLCache(maxsize=1024, ttl=86400 * 7)

OTP: deque[str] = deque(maxlen=10)
OTP_EXPIRE: deque[float] = deque(maxlen=10)

# endregion


# region helpers


def translator(pattern: str) -> Callable[[Translator], Translator]:
    def deco(func: Translator) -> Translator:
        translators.append((regex.compile(pattern, flags=80), func))  # 80: DOTALL | VERBOSE
        return func

    return deco


def helps() -> str:
    return "\n".join(trans[1].__doc__ or "" for trans in translators)


def fusr_to_nfmt_fmt(fusr: str) -> Literal["c", "C", "u", "n", "r", "R"]:
    return cast(
        Literal["c", "C", "u", "n", "r", "R"], {"cn": "c", "CN": "C", "u": "u", "U": "u", "unicode": "u", "ro": "r", "RO": "R"}.get(fusr, fusr)
    )


class _AsyncCleanupQueue:
    _queue: asyncio.Queue | None = None
    _task: asyncio.Task | None = None

    @classmethod
    def schedule(cls, coro):
        if cls._queue is None:
            cls._queue = asyncio.Queue()
            try:
                loop = asyncio.get_running_loop()
                cls._task = loop.create_task(cls._worker())
            except RuntimeError:
                return
        cls._queue.put_nowait(coro)

    @classmethod
    async def _worker(cls):
        if cls._queue is None:
            return
        while True:
            try:
                coro = await cls._queue.get()
                await coro
                cls._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    @classmethod
    async def wait_all_cleanup(cls):
        if cls._queue:
            await cls._queue.join()


def check_otp(otp: str) -> int:
    otp = otp.upper()
    return next(
        (0 if time.monotonic() < exp else 1 for valid, exp in zip(OTP, OTP_EXPIRE) if otp == valid),
        2,  # 0成功1过期2无效
    )


def set_new_otp(exp=60):
    if OTP_EXPIRE and OTP_EXPIRE[-1] > time.monotonic():
        # 已经有正在使用的验证码，直接刷新有效时长，不更新验证码
        OTP_EXPIRE[-1] = time.monotonic() + exp
    else:
        OTP.append("".join(secrets.choice("0123456789ABCDEFGHKLMNPRSTUWXY") for _ in range(6)))
        OTP_EXPIRE.append(time.monotonic() + exp)
    return OTP[-1]


# endregion

# region main


class Tianzi:
    def __init__(
        self, initial_vars: Mapping[str, SupportsStr] | None = None, initial_inner_vars: Mapping[str, Mapping[str, Any]] | None = None
    ):
        self.calc_cache: Cache[Any] = Cache("T.Calc")
        self.result_cache: Cache[SupportsStr] = Cache("T.Ret")
        self.parse_cache: LRUCache[tuple[frozenset[str], frozenset[tuple[str, str]], str], tuple[Translator, Match] | None] = LRUCache(256)
        if initial_vars is not None:
            for k, v in initial_vars.items():
                self.result_cache["Ret":k] = v
        if initial_inner_vars is not None:
            for k1, kv in initial_inner_vars.items():
                for k2, v in kv.items():
                    self.calc_cache[k1:k2] = v
        self.call_stack: list[str] = []
        self.current_stat = _Stat()
        self.nested_inline_epacse: dict[str, str] = {}

        self._sandbox: Sandbox | None = None
        self._sandbox_finalizer: weakref.finalize | None = None

    async def get_sandbox(self) -> Sandbox:
        if self._sandbox is None:
            self._sandbox = await Sandbox.new()

            sandbox = self._sandbox

            def on_gc():
                async def cleanup():
                    with contextlib.suppress(Exception):
                        await sandbox.delete()

                _AsyncCleanupQueue.schedule(cleanup())

            self._sandbox_finalizer = weakref.finalize(self, on_gc)

        return self._sandbox

    # 目前已占用的PUA: 0xE000-0xE07F(ASCII转义) 0xE104-0xE500(嵌套指令打包)
    async def translate(self, text: str, final: bool = False) -> SupportsStr:
        if len(text) > 65535:
            raise BreakOut("[E60.1] 尝试转换的文本过长。输入不得大于65535字符")

        if self.current_stat.allow_pua_warning:
            self.current_stat.pua_warning |= bool(regex.search("[\ue000-\ue07f\ue104-\ue500]", text))
        else:
            self.current_stat.pua_warning = False

        logger.info(f"TRANSLATE {text} ↓↓↓")

        text = regex.sub("\\\\([\x00-\x7f])", lambda m: chr(ord(m.group(1)) + 0xE000), text)

        fields_index: list[tuple[int, int]] = find_outmost_bracket((SYM_HEAD, SYM_TAIL), text)
        fields: deque[SupportsStr] = deque(text[i + len(SYM_HEAD) : j - len(SYM_TAIL)] for i, j in fields_index)
        if not fields:
            ret = text

        nested_cmd_chr_it: Iterator[str] = map(chr, range(0xE104, 0xE500))

        for i, field in enumerate(fields):
            # print(f"debug: {i=} {field=}")
            sfield = str(field)
            nested_cmds: list[tuple[int, int]] = [(0, 0)] + find_outmost_bracket((SYM_HEAD, SYM_TAIL), sfield) + [(len(sfield), 114514)]
            cmd_segs: list[str] = []
            nested_cmd_map: dict[str, str] = {}
            for segi, segj in pairwise(nested_cmds):
                if segi[0] != segi[1]:
                    try:
                        nested_cmd_chr = next(nested_cmd_chr_it)
                    except StopIteration as e:
                        raise PUACharDrained from e
                    nested_cmd_map[nested_cmd_chr] = sfield[segi[0] : segi[1]]
                    cmd_segs.append(nested_cmd_chr)
                cmd_segs.append(sfield[segi[1] : segj[0]])

            sfield = "".join(cmd_segs)
            # print(f"debug: new {sfield=}")
            NIEPACSE_backup = self.nested_inline_epacse.copy()
            self.nested_inline_epacse |= nested_cmd_map
            # print(f"debug: {NIEPACSE_backup=} {self.nested_inline_epacse=}")

            if sfield.startswith(SYN_STRICTMODE):
                cmd_content = sfield[len(SYN_STRICTMODE) :].lstrip()
                if not cmd_content:
                    raise BreakOut("[E30.1] 严式语法必须显式指定翻译器。")
                parts = cmd_content.split()
                cmd_name = parts[0]
                rem_args = parts[1:]
                target_pat = None
                for pat, func in translators:
                    if func.__name__.lower() == cmd_name.lower():
                        target_pat = pat
                        target_func = func
                        break
                else:
                    raise BreakOut(f"[E70.1] 未知的翻译器「{cmd_name}」")

                pargs = {}
                for arg in rem_args:
                    if not arg:
                        continue
                    kv = arg.split("=", 1)
                    if len(kv) <= 1:
                        raise BreakOut("[E30.2] 严式语法必须使用k=v键值对形式显式指定正则字段")
                    k, v = kv
                    pargs[k] = v

                for k in pargs:
                    if k not in target_pat.groupindex:
                        raise BreakOut(f"[E70.2] {k}不是翻译器{cmd_name}的有效正则字段")
                ogd = {name: pargs.get(name) for name in target_pat.groupindex}

                mock_mch = DuckMatch(ogd)

                self.call_stack.append(f"(S){target_func.__name__}[")
                try:
                    fun_t0 = time.perf_counter()
                    field = await target_func(self, mock_mch)
                    fun_t_us = (time.perf_counter() - fun_t0) * 1_000_000
                    if self.call_stack[-1][-1] == "[":
                        self.call_stack[-1] = f"{self.call_stack[-1][:-1]}({fun_t_us:.1f}μs); "
                    else:
                        self.call_stack.append(f"]({fun_t_us:.1f}μs); ")
                except PosteriorReject:
                    raise BreakOut(f"[E70.3] 严式语法下产生的后验拒绝：{cmd_name}<->{ogd}")
                fields[i] = field
                self.nested_inline_epacse = NIEPACSE_backup
                continue

            pcache_key = (
                frozenset(self.result_cache.caches.get("Ret", {}).keys()),
                frozenset(self.nested_inline_epacse.items()),
                sfield,
            )

            if pcache_key in self.parse_cache:
                try:
                    if pch := self.parse_cache[pcache_key]:
                        fun, mch = pch
                        self.call_stack.append(f"(C){fun.__name__}[")
                        fun_t0 = time.perf_counter()
                        field = await fun(self, mch)
                        fun_t_us = (time.perf_counter() - fun_t0) * 1_000_000
                        if self.call_stack[-1][-1] == "[":
                            self.call_stack[-1] = f"{self.call_stack[-1][:-1]}({fun_t_us:.1f}μs); "
                        else:
                            self.call_stack.append(f"]({fun_t_us:.1f}μs); ")
                    else:
                        field = self.epacse(sfield)
                    fields[i] = field
                    self.nested_inline_epacse = NIEPACSE_backup
                    continue
                except PosteriorReject:
                    self.call_stack.pop(-1)

            pr_message = None
            for pat, fun in translators:
                if mch := pat.fullmatch(sfield, concurrent=True, timeout=5):
                    self.call_stack.append(f"{fun.__name__}[")
                    try:
                        fun_t0 = time.perf_counter()
                        field = await fun(self, mch)
                        fun_t_us = (time.perf_counter() - fun_t0) * 1_000_000
                        if self.call_stack[-1][-1] == "[":
                            self.call_stack[-1] = f"{self.call_stack[-1][:-1]}({fun_t_us:.1f}μs); "
                        else:
                            self.call_stack.append(f"]({fun_t_us:.1f}μs); ")
                        self.parse_cache[pcache_key] = fun, mch
                        break
                    except PosteriorReject as pr:
                        pr_message = pr.args
                        self.call_stack.pop()
                        continue
            else:
                if pr_message:
                    return self.breakout(*pr_message)
                field = self.epacse(sfield)
                self.parse_cache[pcache_key] = None
            fields[i] = field
            self.nested_inline_epacse = NIEPACSE_backup

        if not final and fields_index == [(0, len(text))]:
            return fields[0]

        text_lst: list[str] = []
        last_j = 0
        for i, j in fields_index:
            text_lst.extend((text[last_j:i], str(fields.popleft())))
            last_j = j
        text_lst.append(text[last_j:])
        ret = "".join(text_lst)

        if isinstance(ret, str):
            ret = ret.translate(EPACSE)
        logger.info(f"TRANSLATE {text} → {ret} ↑↑↑")

        if final:
            ret = self.epacse(ret)

        return ret

    def check_cache_name(self, cn: str):
        if cn and "_" in cn and not self.current_stat.allow_underscore_in_cache_name:
            return partial(
                self.breakout,
                abbr="[E61缓存名无效]",
                msg="{d} - 缓存名中不允许出现下划线，因其可能导致意外的后果。如果一定要使用下划线，请使用[[config<enable_cache_name_with_underscore>]]。 (E61)",
            )

    def group(self, mch: SupportsGroup, group: str | int) -> str:
        return mch.group(group) or ""

    def egroup(self, mch: SupportsGroup, group: str | int) -> str:
        return (g := mch.group(group)) and self.epacse(g) or ""

    def epacse(self, s: str):
        for c, r in self.nested_inline_epacse.items():
            s = s.replace(c, r)
        return s.translate(EPACSE)

    async def tegroup(self, mch: SupportsGroup, group: str | int) -> SupportsStr:
        return await self.translate(self.egroup(mch, group))

    async def stegroup(self, mch: SupportsGroup, group: str | int) -> str:
        return str(await self.tegroup(mch, group))

    def breakout(self, mch: SupportsGroup, abbr: str, msg: str) -> str:
        match self.current_stat.err_level:
            case "ignore":
                return ""
            case "abort":
                return self.egroup(mch, 0)
            case "inline":
                return abbr
            case "raise":
                raise BreakOut(msg.replace("{d}", self.egroup(mch, 0)))


# endregion

# region translators


# region Nocensor
@translator("nocensor\\s*(?P<otp>([0-9A-Z]{6}))?")
async def Nocensor(self: Tianzi, mch: SupportsGroup) -> SupportsStr:
    """临时关闭屏蔽词系统"""
    logger.info(f"Nocensor ← {mch.groupdict()} debug: {OTP=} {OTP_EXPIRE=}")
    otp = self.group(mch, "otp")
    if not otp:
        return self.breakout(
            mch,
            "[E72.1验证码缺失]",
            f"{{d}} - 需要验证码才能执行此操作。当前验证码为【{(val_otp:=set_new_otp())}】。请使用[[nocensor {val_otp}]]。 (E72.1)\n警告：使用此指令造成的一切后果由调用者承担，本机不负任何责任！",
        )
    if (chk := check_otp(otp)) == 0:
        self.current_stat.censor = False
        return ""
    elif chk == 1:
        return self.breakout(mch, "[E72.2验证码过期]", f"{{d}} - 此验证码已过期。当前验证码为【{set_new_otp()}】。 (E72.2)")
    elif chk == 2:
        return self.breakout(mch, "[E72.3验证码无效]", f"{{d}} - 此验证码无效。当前验证码为【{set_new_otp()}】。 (E72.3)")
    else:
        raise WhatTheFuckIsThis


# region Reset
@translator("RST")
async def Reset(self: Tianzi, _: SupportsGroup) -> SupportsStr:
    """重置 「Reset」
    语法：RST"""
    logger.info("Reset")
    self.calc_cache.clear()
    self.result_cache.clear()
    self.parse_cache.clear()
    for lex in lexloader.LOAD_CACHE.values():
        lex.schemas.qcache.clear()
    self.nested_inline_epacse.clear()
    return ""


# region LingerAssign
@translator(f"(?P<val>.*?){RSYN_LA}(?P<cname>.+)")
async def LingerAssign(self: Tianzi, mch: SupportsGroup) -> SupportsStr:
    """驻留赋值 「LingerAssign」
    语法：{变量名}>>>{驻留名}"""
    varnames = await self.stegroup(mch, "val")
    linger_name = self.egroup(mch, "cname")
    linger_names: set[str] = set()
    if varnames:
        for varname in varnames.split():
            linger_names.add(varname)
    else:
        linger_names |= self.result_cache.caches["Ret"].keys()
    global LINGERS
    LINGERS[linger_name] = {
        varname: self.result_cache.caches["Ret"][varname] for varname in linger_names if varname in self.result_cache.caches["Ret"]
    }
    return ""


# region LingerRef
@translator(f"{RSYN_LR}(?P<cname>.+)")
async def LingerRef(self: Tianzi, mch: SupportsGroup) -> SupportsStr:
    """驻留引用 「LingerRef」
    语法：<<<{驻留名}"""
    lingername = self.egroup(mch, "cname")
    if lingername not in LINGERS.keys():
        return self.breakout(mch, "[E71.1驻留名无效]", f"{{d}} - 驻留名{lingername}不存在。 (E71.1)")
    self.result_cache.caches["Ret"].update(LINGERS[lingername])
    return ""


# region Slice
@translator(
    f"(?P<target>.+?){RSYN_SLICESEP}(?P<start>[\\-0-9{RCS_INLINE}]+)?{RSYN_SLICESEP}(?P<stop>[\\-0-9\\@{RCS_INLINE}]+)?({RSYN_SLICESEP}(?P<step>[\\-0-9{RCS_INLINE}]+))?"
)
async def Slice(self: Tianzi, mch: SupportsGroup) -> SupportsStr:
    """切片 「Slice」
    语法：{值}:{起始}:{结束}[:步长]"""
    logger.info(f"Slice ← {mch.groupdict()}")
    main = await self.stegroup(mch, "target")
    _start = cast(str, await self.tegroup(mch, "start")) or None
    _stop = cast(str, await self.tegroup(mch, "stop")) or None
    _step = cast(str, await self.tegroup(mch, "step")) or None
    try:
        if _stop == "@":
            if not _start:
                raise PosteriorReject
            start = int(_start)
            ret = main[start]
        else:
            if not _start and not _stop and not _step:
                raise PosteriorReject
            start = _start and int(_start)
            stop = _stop and int(_stop)
            step = _step and int(_step)
            ret = main[start:stop:step]
    except ValueError:
        return self.breakout(mch, "[E73.11切片参数无效]", "{d} - 切片参数无效。 (E73.11)")
    except IndexError:
        return self.breakout(mch, "[E73.21切片越界]", "{d} - 切片越界。 (E73.21)")
    logger.info(f"Slice → {ret}")
    return ret


# region Config
@translator("config<(?P<item>.+)>")
async def Config(self: Tianzi, mch: SupportsGroup) -> SupportsStr:
    """配置项 「Config」
    语法：config<配置项>"""
    logger.info(f"Config ← {mch.groupdict()}")

    item = self.egroup(mch, "item")

    match item.lower():
        case "enable_cache_name_with_underscore" | "vu":
            self.current_stat.allow_underscore_in_cache_name = True
        case "disable_cache_name_with_underscore":
            self.current_stat.allow_underscore_in_cache_name = False
        case "error_level_ignore" | "eig":
            self.current_stat.err_level = "ignore"
        case "error_level_abort" | "ea":
            self.current_stat.err_level = "abort"
        case "error_level_inline" | "eil":
            self.current_stat.err_level = "inline"
        case "error_level_raise" | "er":
            self.current_stat.err_level = "raise"
        case "max_calculate_output_normal":
            self.current_stat.max_calc_output_length = 512
        case "max_calculate_output_long" | "cl":
            self.current_stat.max_calc_output_length = 1024
        case "max_calculate_output_unlimited" | "cu":
            self.current_stat.max_calc_output_length = 1145141919810
        case "disable_pua_warning":
            self.current_stat.allow_pua_warning = False
        case "enable_pua_warning":
            self.current_stat.allow_pua_warning = True
        case "enable_calc_bignum" | "cb":
            self.current_stat.allow_calc_big_number = True
        case "disable_calc_bignum":
            self.current_stat.allow_calc_big_number = False
        case _:
            return self.breakout(mch, "[E41配置无效]", f"{{d}} - {item} 不是有效的配置项名称。 (E41)")

    return ""


# region 彩蛋
@translator(
    "".join(chr(ord(c) - 42) for c in "畉咧〫宱寃叴个刱皮绲枫签桲"),
)
async def TheAnswer(self: Tianzi, _: SupportsGroup) -> SupportsStr:
    """..."""
    return ANSWER_TO_THE_ULTIMATE_QUESTION_OF_LIFE_THE_UNIVERSE_AND_EVERYTHING.ANSWER


# region Calculate
@translator(f"{RSYN_CALC}\\s?(?P<expr>.+)")
async def Calculate(self: Tianzi, mch: SupportsGroup) -> SupportsStr:
    """计算 「Calculate」
    语法：={表达式}"""

    logger.info(f"Calculate ← {mch.groupdict()}")

    main = self.egroup(mch, "expr")
    cache_vars = {
        k: (v.tolist() if isinstance(v, (np.ndarray, awkward.Array, np.generic)) else v)
        for k, v in self.result_cache.caches.get("Ret", {}).items()
    }

    inner_cache_vars = {
        field: SimpleNamespace(
            **{
                name: (
                    v.tolist()
                    if isinstance(
                        (v := self.calc_cache.get(field, name)),
                        (np.ndarray, awkward.Array, np.generic),
                    )
                    else v
                )
                for name in names
            }
        )
        for field, names in self.calc_cache.caches.items()
    }

    print(f"debug: {repr(cache_vars)} {repr(inner_cache_vars)}")

    class FFSEval(simpleeval.SimpleEval):
        @staticmethod
        def _raise(*_, **_____) -> Never:
            raise simpleeval.InvalidExpression

        @no_type_check
        def __init__(self, names=None):
            super().__init__()
            self.builtins = {}
            self.nodes[ast.Call] = self._raise
            self.nodes[ast.ListComp] = self._raise
            self.nodes[ast.SetComp] = self._raise
            self.nodes[ast.DictComp] = self._raise
            self.nodes[ast.GeneratorExp] = self._raise
            self.attrfilter = self._attr_filter
            if names is not None:
                self.names = names

        @staticmethod
        def _attr_filter(_, attr):
            if attr.startswith("_"):
                raise simpleeval.InvalidExpression

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", simpleeval.AssignmentAttempted)
            try:
                safe_cache_vars = {
                    name: value for name, value in cache_vars.items() if type(value) in {int, float, str, bool, NoneType, complex, bytes}
                }  # 有逃逸风险的直接去掉，尝试访问的话从NameError跳出到pysb就行
                ev = FFSEval(safe_cache_vars)
                retdirect: SupportsStr = ev.eval(main)
                logger.info(f"{main} → FFSEval")
            except simpleeval.AssignmentAttempted:
                raise simpleeval.InvalidExpression
    except simpleeval.InvalidExpression as ieexp:
        if not self.current_stat.allow_calc_big_number and isinstance(ieexp, simpleeval.NumberTooHigh):
            return self.breakout(
                mch,
                "[E73.24数值过大]",
                "{d} - 计算错误：数值过大。如果需要在沙盒中重试，请指定[[config<enable_calc_bignum>]] (E73.24)",
            )
        logger.info(f"{main} → PySB")

        try:
            sandbox = await self.get_sandbox()
        except Exception as e:
            return self.breakout(
                mch,
                f"[E24.1,{type(e).__name__}计算失败]",
                f"{{d}} - 沙盒服务器错误：{type(e).__name__} - {e}；请联系找北 (E24.1,{type(e).__name__})",
            )

        output = await sandbox.run(main, cache_vars | inner_cache_vars)
        logger.info(f"Calc Server output: {output}")

        if output.err.msg or output.err.typ:
            return self.breakout(mch, f"[E24.2,{output.err.typ}计算错误]", f"{{d}} - 计算错误：{output.err.msg} (E24.2,{output.err.typ})")

        retdirect = output.retstr if (output.retval is RETVAL_MISSING) or (output.retval is None) else output.retval

        self.result_cache.caches["Ret"].update(output.variables.valids)
        self.result_cache.caches["ReadOnly"] = cast(dict[str, SupportsStr], output.variables.invalids)
        for iv in output.variables.invalids:
            self.result_cache.caches["Ret"].pop(iv, None)

    except KeyboardInterrupt, SystemExit:
        raise
    except BaseException as e:
        traceback.print_exc()
        return self.breakout(mch, f"[E14,{type(e).__name__}计算错误]", f"{{d}} - 计算错误：{e} (E14,{type(e).__name__})")

    if (lsr := len(str(retdirect))) > self.current_stat.max_calc_output_length:
        return self.breakout(mch, "[E60.4结果过长]", f"{{d}} - 计算结果过长（{lsr}字符）。(E60.4)")

    logger.info(f"Calculate → {retdirect}")
    return retdirect


# region InnerValRef
@translator(f"{RSYN_IVR}((?P<field>.+?){RSYN_IVFIELD})?(?P<cname>.+)")
async def InnerValRef(self: Tianzi, mch: SupportsGroup) -> SupportsStr:
    """内部值引用 「InnerValRef」
    语法：<<[{作用域}:]{缓存名}"""

    logger.info(f"InnerValRef ← {mch.groupdict()}")

    field: str = await self.stegroup(mch, "field")
    name: str = await self.stegroup(mch, "cname")

    if bo := self.check_cache_name(name):
        return bo(mch)

    logger.info(f"caches: {self.calc_cache.caches}")

    for _, trans in translators:
        _field = field or trans.__name__
        if ret := self.calc_cache.get(_field, name, None):
            ret = str(ret)
            logger.info(f"InnerValRef → {ret}")
            return ret
        if field:
            break
    ret = self.egroup(mch, 0)
    logger.info(f"InnerValRef → {ret}")
    return ret


# region Reference
@translator(f"{RSYN_REF}(?P<cname>.+)")
async def Reference(self: Tianzi, mch: SupportsGroup) -> SupportsStr:
    """引用 「Reference」
    语法：<{缓存名}"""
    logger.info(f"Reference ← {mch.groupdict()}")
    name = await self.stegroup(mch, "cname")
    ret = self.result_cache.get("Ret", name, self.result_cache.get("ReadOnly", name))
    if ret is None:
        raise PosteriorReject
    logger.info(f"Reference → {ret}")
    return ret


# region InnerValAssign
@translator(f"(?P<val>.+?){RSYN_IVA}((?P<field>.+?){RSYN_IVFIELD})?(?P<cname>[^{RSYM_CACHE}]+?)(?P<output>{RSYM_CACHE}?)")
async def InnerValAssign(self: Tianzi, mch: SupportsGroup) -> SupportsStr:
    """内部值赋值 「InnerValAssign」
    语法：{值}>>{作用域}:{缓存名}[>]"""
    logger.info(f"InnerValAssign ← {mch.groupdict()}")
    field: str = await self.stegroup(mch, "field")

    if not field:
        return self.breakout(mch, "[E71.2作用域缺失]", "在为内部值赋值的时候，必须指定作用域。(E71.2)")

    val: SupportsStr = await self.tegroup(mch, "val")
    name: str = await self.stegroup(mch, "cname") or ""
    output: bool = bool(self.group(mch, "output"))

    if bo := self.check_cache_name(name):
        return bo(mch)

    self.calc_cache[field:name] = val

    return val if output else ""


# region Font
@translator(f"""
    (?P<target>.+?)\\s?
    {RSYN_FONT}
    (?P<font>[^{RSYM_TAIL}{RCS_SEP}{RCS_SEGSEP}]+?)
    ({RSYM_MODIFY}
        (?P<charset>.+?)
    )?
    (\\s?{RSYM_CACHE}
        (?P<cname>.+?)
    )?
""")
async def Font(self: Tianzi, mch: SupportsGroup) -> SupportsStr:
    """字体 「Font」
    语法：{字符串} :{字体}[>{缓存名}]"""

    logger.info(f"Font ← {mch.groupdict()}")

    font_name: str = await self.stegroup(mch, "font")

    font: dict[int, int] | dict[int | str, int | str] | dict[str, str] | None = fonts.get(font_name.lower(), None)
    if font is None:
        raise PosteriorReject(mch, "[E42字体无效]", f"{{d}} - 「{font_name}」不是有效的字体名称。(E42)")

    cache_name: str = await self.stegroup(mch, "cname")
    data: str = await self.stegroup(mch, "target")
    charset: str = await self.stegroup(mch, "charset")
    if bo := self.check_cache_name(cache_name):
        return bo(mch)

    for k, v in font.items():
        if (not isinstance(k, str)) or (charset and (k not in charset)):
            continue
        data = data.replace(k, cast(str, v))
    # 此处的cast是为了直接复用带str:str的翻译表

    ret = data.translate({k: v for k, v in font.items() if isinstance(k, int) and (not charset or chr(k) in charset)})
    self.result_cache["Ret":cache_name] = ret
    logger.info(f"Font → {ret}")

    return ret


# region Range
@translator(f"""
    (?P<left>[^{RSYM_HEAD}{RSYM_TAIL}{RSYM_MODIFY}{RSYN_CALC}\\s]+?)
    \\s*(?P<sep>([{RCS_RANGE}]|——))\\s*
    (?P<right>[^{RSYM_HEAD}{RSYM_TAIL}{RSYM_MODIFY}\\s:]+?(?<!({RCG_ELLIPSIS})))
    (
        (:
            (?P<fspec>(?!(cn|CN|u|U|unicode|nul|ro|RO))
                (
                    (?P<fill>[^{{}}<>;]?)
                    (?P<align>[<>=^])
                )?
                (?P<sign>[-+ ]?)
                \\#?
                0?
                [0-9]*
                [_,]?
                (\\.(?P<perc>[0-9]*))?
                (?P<ftype>[bcdeEfFgGnosxX%]?)
            )?
            (
                [{RCS_SEGSEP}]?
                (?P<fusr>(cn|CN|u|U|unicode|nul|ro|RO))
            )?
        )?
        (\\s?{RSYM_MODIFY}
            (?P<rand>({"|".join(map(escape,rngs.names))}))
        )?
        (\\s?[{RSYM_CACHE}]
            (?P<cname>.+?)
        )?
    ){{3}}""")
async def Range(self: Tianzi, mch: SupportsGroup) -> SupportsStr:
    """随机数 「Range」
    语法：{下界}-{上界}[:[{Python式格式};]{转换格式}][@{分布}][>{缓存名}]"""

    logger.info(f"Range ← {mch.groupdict()}")

    cache_name: str = await self.stegroup(mch, "cname")
    rand_type: str = await self.stegroup(mch, "rand")

    if bo := self.check_cache_name(cache_name):
        return bo(mch)

    cached_rand: float | None = self.calc_cache.get("Range", cache_name, None) if cache_name else None
    rand: float = rngs.get(rand_type, random.random)() if cached_rand is None else cached_rand
    self.calc_cache["Range":cache_name] = rand

    lit_value1, lit_value2 = (await self.tegroup(mch, "left")), (await self.tegroup(mch, "right"))
    try:
        lv1v = isinstance(lit_value1, Value)
        lv2v = isinstance(lit_value2, Value)
        if lv1v and lv2v:
            v_ftype = ""
            v_otype = "n"
            value1, value2 = lit_value1, lit_value2
        elif lv1v:
            value1 = lit_value1
            v_ftype, v_otype, value2, _ = numify(str(lit_value2))
        elif lv2v:
            v_ftype, v_otype, value1, _ = numify(str(lit_value1))
            value2 = lit_value2
        else:
            v_ftype, v_otype, (value1, value2), _ = numsify((str(lit_value1), str(lit_value2)))
    except OverflowError:
        if rand_type:
            return self.breakout(mch, "[E83尚未支持]", "{d} - 尚不支持在分布上进行无界随机，还望谅解。(E83)")
        return self.breakout(mch, "[E73.33边界无效]", "{d} - 边界不可以是Inf或NaN。(E73.12)")
    except ValueError:
        if self.egroup(mch, "sep") == "-":
            # 一般是带有负数的choice被识别成Range了，此时抛个后验拒绝让choice能吃到
            raise PosteriorReject
        return self.breakout(mch, "[E73.13非数值]", f"{{d}} - 输入的边界「{lit_value1}」「{lit_value2}」无法被解析为范围。(E73.13)")

    c_ftype: str = self.egroup(mch, "ftype")
    c_otype: str = self.egroup(mch, "fusr")

    ftype = c_ftype or v_ftype or "s"
    otype = c_otype or v_otype or "nul"

    cf_fspec: str = self.egroup(mch, "fspec") or ""
    cf_fill: str = self.egroup(mch, "fill") or ""
    cf_align: str = self.egroup(mch, "align") or ""
    cf_sign: str = self.egroup(mch, "sign") or ""
    cf_perc: int | None = int(eperc) if (eperc := self.egroup(mch, "perc")) else None

    logger.info(
        f"Range: {lit_value1=} {lit_value2=} {value1=} {value2=} {v_ftype=} {v_otype=} {ftype=} {otype=} {cf_fspec=} {cf_fill=} {cf_align=} {cf_sign=} {cf_perc=}"
    )

    if isinstance(value1, int) and isinstance(value2, int):
        chosen = value1 + int((value2 - value1 + 1) * rand)
    else:
        chosen = value1 + (value2 - value1) * rand

    chosen: Value = numsimp(chosen)

    self.result_cache["Ret":cache_name] = chosen

    if cf_fspec or (otype != "nul"):
        try:
            ret: str | Value = numfmt(
                fusr_to_nfmt_fmt(otype),
                cf_fspec,
                cf_fill,
                cf_align,
                cf_sign,
                cf_perc,
                ftype,
                chosen,
            )
        except UnicodeError:
            return self.breakout(
                mch, "[E73.35码点无效]", f"{{d}} - 随机结果「{chosen}」不是一个有效的Unicode码点。对于u模式，上下界不应超出0~1114111。(E73.35)"
            )
        except ValueError:
            return self.breakout(
                mch,
                "[E74.3格式无效]",
                f"{{d}} - 输入/推定的格式 {cf_fspec or "NUL"} -> {otype} 不可用于格式化「{chosen}」（{type(chosen).__name__}）。(E74.3)",
            )
    else:
        ret = chosen

    logger.info(f"Range → {ret}")

    return ret


# region ImmediateNumbers
@translator(f"{RSYN_IN}(?P<val>.+?)([:;](?P<fmt>(c|cn|CN|u|U|unicode|n|norm|normal|ro|RO|x|X|d|f)))?")
async def ImmediateNumbers(self: Tianzi, mch: SupportsGroup) -> SupportsStr:
    """立即数 「ImmediateNumbers」
    语法：#{数值}"""
    logger.info(f"ImmediateNumbers ← {mch.groupdict()}")

    _fmt = await self.stegroup(mch, "fmt")
    fmt: NSLVL | None = {
        "c": NSLVL.C,
        "cn": NSLVL.C,
        "CN": NSLVL.C,
        "u": NSLVL.U,
        "U": NSLVL.U,
        "unicode": NSLVL.U,
        "n": NSLVL.N,
        "norm": NSLVL.N,
        "normal": NSLVL.N,
        "ro": NSLVL.R,
        "RO": NSLVL.R,
        "x": NSLVL.X,
        "hex": NSLVL.X,
        "X": NSLVL.XU,
        "HEX": NSLVL.XU,
        "d": NSLVL.N,
        "f": NSLVL.N,
        None: None,
        "": None,
    }[_fmt]

    val = await self.tegroup(mch, "val")

    if (fmt == "n") and isinstance(val, numbers.Number):
        ret = val
    else:
        try:
            ret = numify(str(val), lvl=fmt)[2]
        except ValueError:
            return self.breakout(mch, "[E36.2数值无效]", f"{{d}} - {_fmt or "默认/语法糖"}模式下无效的数值。(E36.2)")

    logger.info(f"ImmediateNumbers → {ret}")
    return ret


# region Format
@translator(f"""
    (?P<target>
        [^{RSYM_MODIFY}\\s\\?]+?
        (?<!({RCG_ELLIPSIS}))
    )
    (:
        (?P<fspec>(?!(cn|CN|u|U|unicode|ro|RO))
            (
                (?P<fill>[^{{}}<>;]?)
                (?P<align>[<>=^])
            )?
            (?P<sign>[-+ ]?)
            \\#?
            0?
            [0-9]*
            [_,]?
            (\\.(?P<perc>[0-9]*))?
            (?P<ftype>[bcdeEfFgGnosxX%]?)
        )?
        (
            [{RCS_SEGSEP}]?
            (?P<fusr>(cn|CN|u|U|unicode|ro|RO))
        )?
    )
    (\\s?[{RSYM_CACHE}]
        (?P<cname>.+?)
    )?
""")
async def Format(self: Tianzi, mch: SupportsGroup) -> SupportsStr:
    """格式化 「Format」
    语法：{数据}[:[{Python式格式};]{转换格式}][>{缓存名}]"""

    logger.info(f"Format ← {mch.groupdict()}")

    cache_name: str = await self.stegroup(mch, "cname")

    if bo := self.check_cache_name(cache_name):
        return bo(mch)

    data: str = await self.stegroup(mch, "target")
    if not data:
        return ""
    cf_fspec: str = self.egroup(mch, "fspec") or ""
    cf_fill: str = self.egroup(mch, "fill") or ""
    cf_align: str = self.egroup(mch, "align") or ""
    cf_sign: str = self.egroup(mch, "sign") or ""
    cf_perc: int | None = int(eperc) if (eperc := self.egroup(mch, "perc")) else None
    c_ftype: str = self.egroup(mch, "ftype")
    c_otype: str = self.egroup(mch, "fusr") or "nul"

    try:
        v_ftype, v_otype, value, _ = numify(data, True)
    except TypeError, ValueError:
        return self.breakout(mch, "[E36.3解析不能]", f"{{d}} - 输入的值「{data}」无法被解析为数值。(E36.3)")

    value = numsimp(value)

    ftype = c_ftype or v_ftype or "s"
    otype = c_otype or v_otype or "nul"

    try:
        ret: str = numfmt(
            fusr_to_nfmt_fmt(otype),
            cf_fspec,
            cf_fill,
            cf_align,
            cf_sign,
            cf_perc,
            ftype,
            value,
        )
    except UnicodeError:
        return self.breakout(
            mch, "[E42.2码点无效]", f"{{d}} - 输入值「{value}」不是一个有效的Unicode码点。对于u模式，码点数值不应超出0~1114111。(E42.2)"
        )
    except ValueError:
        return self.breakout(
            mch,
            "[E33.2格式无效]",
            f"{{d}} - 输入/推定的格式 {cf_fspec or "NUL"} -> {otype} 不可用于格式化「{value}」（{type(value).__name__}）。(E33.2)",
        )

    self.result_cache["Ret":cache_name] = ret
    logger.info(f"Format → {ret}")

    return ret


# region Repeat
@translator(
    f"(?P<target>[^{RSYN_REPEAT}{RSYN_REPEATL}]+?)(?P<ast>{RSYN_REPEATL}|{RSYN_REPEAT})(?P<num>[^{RSYM_TAIL}]+?)(\\s?{RSYM_CACHE}(?P<loopvar>[^{RSYM_TAIL}]+?))?(\\s?{RSYM_MODIFY}(?P<offset>[^{RSYM_TAIL}]+?))?(?P<trim>\\.\\.(\\.|\\?|\\!))?"
)
async def Repeat(self: Tianzi, mch: SupportsGroup) -> SupportsStr:
    """重复 「Repeat」\
    语法：{字符串} *[*]{次数}"""
    lazy = mch.group("ast") == SYN_REPEATL
    loopvar: str = self.egroup(mch, "loopvar")
    trim: str = self.group(mch, "trim")
    trim = trim and trim[2:]
    try:
        offset: int = int(cast(str, ((await self.tegroup(mch, "offset")) or 0)))
    except ValueError:
        return self.breakout(mch, "[E36.5解析不能]", f"{{d}} - 偏移量「{self.egroup(mch, 'offset')}」无法被解析为整数。(E36.5)")

    _head = self.egroup(mch, "target")

    logger.info(f"Repeat ←  {lazy=} {loopvar=}")

    def __sepsep_punct(head: str) -> tuple[str, str]:
        _mch = regex.fullmatch(f"(?P<main>[\\s\\S]+?)(?P<sep>[{RCS_SEP}]+)?", head)
        if not _mch:
            raise PosteriorReject
        logger.info(f"Repeat SSP ← {_mch.groupdict()}")
        return (_mch.group("main") or ""), (_mch.group("sep") or "")

    def __sepsep_tail(head: str) -> tuple[str, str]:
        _end = head.rfind(SYM_TAIL[-1])
        logger.info(f"Repeat SST ← {_end=}")
        return (head[: _end + 1], _head[_end + 1 :]) if _end != -1 else (_head, "")

    match trim:
        case "!":  # ……..!表示仅标点推断
            main, sep = __sepsep_punct(_head)
        case ".":  # ……...表示不去尾随
            main, sep = _head, ""
        case "?":  # ……..?表示仅自动推断
            main, sep = __sepsep_tail(_head)
        case _:  # 全自动推断
            main, sep = __sepsep_punct(_head)
            if not sep:
                main, sep = __sepsep_tail(_head)

    try:
        num: int = int(cast(str, (await self.tegroup(mch, "num"))))
    except ValueError:
        return self.breakout(mch, "[E36.4解析不能]", f"{{d}} - 次数「{self.egroup(mch, 'num')}」无法被解析为整数。(E36.4)")

    logger.info(f"Repeat ← {num=}")

    if num > 4096:
        return self.breakout(mch, "[E22.1次数无效]", f"{{d}} - 次数「{num}」超出范围。最大次数为4096。(E22.2)")
    if len(main) * num > 131072:
        return self.breakout(
            mch,
            "[E22.2次数无效]",
            f"{{d}} -次数「{num}」超出范围。在该输入长度（{len(main)}字符）下，最多重复次数为{131072//len(main)}次。(E22.2)",
        )

    if lazy:
        ret_lst = []
        for lv in range(offset, num + offset):
            self.result_cache["Ret":loopvar] = lv
            ret_lst.append(str(await self.translate(main)))
    else:
        self.result_cache["Ret":loopvar] = offset
        ret_lst = repeat(str(await self.translate(main)), num)
        self.result_cache["Ret":loopvar] = offset + num - 1
    ret = sep.join(ret_lst)

    logger.info(f"Repeat → {ret}")
    return ret


# region Choice
@translator(f"""
    (?P<main>[^>=]+?(?P<sep>[{RCS_SPLITSEP}])[^>=]+(?:(?P=sep)[^>=]+)*)
    (
        (\\s?{RSYM_MODIFY} (?P<rand>({"|".join(rsgs.names)})) )?
        (\\s?{RSYM_CACHE} (?P<cname>[^{RSYM_LENGTH}{RSYM_TAIL}]+?) )?
        (\\s?{RSYM_LENGTH} (?P<count>.+?) )?
    ){{3}}
""")
async def Choice(self: Tianzi, mch: SupportsGroup) -> SupportsStr:
    """选择 「Choice」 语法：{选项1} {选项2} {选项3}...[>{缓存名}][@{分布}][={选择数}]"""
    logger.info(f"Choice ← {mch.groupdict()}")
    cache_name: str = await self.stegroup(mch, "cname")
    if bo := self.check_cache_name(cache_name):
        return bo(mch)
    length: int = max(1, int(cast(str, await self.tegroup(mch, "count")) or 1))
    rand_type: str = await self.stegroup(mch, "rand")

    sep: str = self.group(mch, "sep") or ""
    main: str = self.group(mch, "main") or ""
    splitted = [self.epacse(seg) for seg in main.split(sep) if seg]
    logger.info(f"Choice ↔ {splitted}")

    if len(splitted) <= 1:
        raise PosteriorReject
    length = min(length, len(splitted))
    _cache_name = f"{cache_name}${length}"

    weights = None
    if cache_name:
        weights: list[float] | None = self.calc_cache.get("Choice", _cache_name)
        if weights is None and length == 1:
            _rand: float | None = self.calc_cache.get("Range", cache_name)
            if _rand is not None:
                weights = [0.0] * len(splitted)
                weights[int(_rand * len(splitted))] = 1.0
    if weights is None:
        if rand_type:
            weights = list(rsgs[rand_type](len(splitted)))
        else:
            weights = [1 / len(splitted)] * len(splitted)

    options: list[str] = []
    opt_weights: list[float] = []
    for item in splitted:
        _parts = item.split(SYM_MODIFY, 1)
        if len(_parts) > 1:
            try:
                opt_weights.append(float(_parts[1]))
            except ValueError:
                raise PosteriorReject
        else:
            opt_weights.append(1.0)
        options.append(_parts[0])
    weights = [w * ow for w, ow in zip(weights, opt_weights)]
    swght = sum(weights)
    weights = [w / swght for w in weights]

    if cache_name:
        self.calc_cache["Choice":_cache_name] = weights

    chosens: list[str] = list(nrandom.choice(options, length, False, weights))
    chosens = [str(await self.translate(opt)) for opt in chosens]
    ret = sep.join(chosens)
    self.result_cache["Ret":cache_name] = ret
    logger.info(f"Choice → {ret}")
    return ret


# region Lex
@translator(f"""
    (?P<lex>({"|".join(escape(l) for l in lexloader.all_lexicons)}))
    (\\.(?P<colname>[^\\({RSYM_CACHE}]+?))?
    (\\{{(?P<query>.+?)\\}})?
    ({RSYM_CACHE}(?P<cname>.+))?
    """)
async def Lex(self: Tianzi, mch: SupportsGroup) -> SupportsStr:
    """词库 「Lex」
    语法：{词库名}[.{列名}][{ {查询} }]"""
    logger.info(f"Lex ← {mch.groupdict()}")

    lex_name: str = await self.stegroup(mch, "lex")
    colname: str = await self.stegroup(mch, "colname")
    query: str = await self.stegroup(mch, "query")
    cache_name: str = await self.stegroup(mch, "cname")
    if lex_name not in lexloader.all_lexicons:
        if colname or query:
            return self.breakout(mch, "[E66词库不存在]", f"{{d}} - 没有名为「{lex_name}」的词库")
        raise PosteriorReject
    lex = await lexloader.Lexicon.load(lex_name)

    if self.calc_cache.get("Lex", cname := f"{lex_name}_{cache_name}") is not None:
        ret = self.calc_cache["Lex":cname]
        print("debug: lexret:", ret)
        try:
            ret = int(ret)
        except ValueError:
            return self.breakout(mch, "[E01填词失败]", "{d} - Lex的内部值必须为int或可以为int。 (E01.1)")

    else:
        if bo := self.check_cache_name(cache_name):
            return bo(mch)
        try:
            ret = lex.schemas.query_pop(query)
        except Exception as e:
            traceback.print_exc()
            return self.breakout(mch, "[E01填词失败]", f"{{d}} - 填词失败：{repr(e)} (E01)")

    if ret is None:
        return self.breakout(mch, "[E62填词失败]", "{d} - 没有符合条件的词。(E62)")

    if cache_name:
        self.calc_cache["Lex":f"{lex_name}_{cache_name}"] = ret

    retstr = lex[colname:ret]

    self.result_cache["Ret" : self.egroup(mch, "cache_name")] = retstr
    logger.info(f"Lex → {repr(retstr)} debug: {self.calc_cache.caches}")
    return retstr


# region Assign
@translator(f"(?P<val>[^>]+?){RSYM_CACHE}(?P<cname>[^{RSYM_CACHE}]+)(?P<output>{RSYM_CACHE}?)")
async def Assign(self: Tianzi, mch: SupportsGroup) -> SupportsStr:
    """赋值 「Assign」
    语法：{值}>{缓存名}[>]"""
    logger.info(f"Assign ← {mch.groupdict()}")
    val = await self.tegroup(mch, "val")
    cache_name = await self.stegroup(mch, "cname")

    if bo := self.check_cache_name(cache_name):
        return bo(mch)

    self.result_cache["Ret":cache_name] = val
    return val if self.group(mch, "output") else ""

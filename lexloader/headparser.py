import re
from typing import NamedTuple

from frozendict import frozendict

type Group = dict[str, set[str] | None]

type FrozenGroup = frozendict[str, frozenset[str] | None]


class ColSpec[G: Group | FrozenGroup | None](NamedTuple):
    group: G
    name: str
    proto: str


type TColSpec = ColSpec[Group]
type TColSpecN = ColSpec[Group | None]

type TFrozenColSpec = ColSpec[FrozenGroup]
type TFrozenColSpecN = ColSpec[FrozenGroup | None]


class Head[M: TColSpec | TColSpecN](NamedTuple):
    main: M
    jit: list[TColSpecN]


type THead = Head[TColSpec]
type THeadN = Head[TColSpecN]


class ColIDT(NamedTuple):
    name: str
    proto: str


class CompilingHeads(NamedTuple):
    main: ColIDT
    aot: list[ColIDT]


PGroup = re.compile(r"(?P<group>[^;{}()!]+)(?:!(?P<schema>[^;{}()]+|\([^)]*\)))?")

PCol = re.compile(r"(?:\{(?P<group>[^}]*)\})?(?P<name>[^;{}@:.\[\]]+)\.(?P<proto>[^;{}@:.\[\]]+)(?:\[(?P<jit>[^\]]*)\])?")

Pidt = re.compile(r"(?:\{[^}]*\})?(?P<name>[^;{}@:.\[\]]+)\.(?P<proto>[^;{}@:.\[\]]+)")


def parse_group(content: str) -> Group:
    if not content:
        return {}
    result: Group = {}
    for m in PGroup.finditer(content):
        gname: str = m.group("group")
        schema: str | None = m.group("schema")
        if not schema:
            result[gname] = None
        elif schema.startswith("(") and schema.endswith(")"):
            result[gname] = set(schema[1:-1].split(";"))
        else:
            result[gname] = {schema}
    return result


def split_om(s: str) -> list[str]:
    parts: list[str] = []
    bracket = brace = paren = False
    i0 = 0
    for i, c in enumerate(s):
        if c == "[":
            bracket = True
        elif c == "]":
            bracket = False
        elif c == "{":
            brace = True
        elif c == "}":
            brace = False
        elif c == "(":
            paren = True
        elif c == ")":
            paren = False
        elif c == ";" and not (bracket or brace or paren):  # 能复用textutil但是不想复用……详见schemas
            parts.append("".join(s[i0:i]))
            i0 = i + 1
    if i0 < len(s):
        parts.append(s[i0:])
    return parts


def parse_idt(s: str) -> ColIDT:
    if m := Pidt.match(s):
        return ColIDT(m.group("name"), m.group("proto"))
    else:
        raise SyntaxError


def compile_get_parts(s: str) -> tuple[str, list[str]]:
    parts = s.split("@")
    return parts[0], (split_om(parts[1]) if len(parts) > 1 else [])


def compile_parse(s: str) -> CompilingHeads:
    parts = s.split("@")
    main_part = parts[0]
    aot_part = parts[1] if len(parts) > 1 else None

    main = parse_idt(main_part)

    aot: list[ColIDT] = []
    if aot_part:
        aot.extend(parse_idt(col_str) for col_str in split_om(aot_part))
    return CompilingHeads(main, aot)


def parse_jit(section: str | None) -> list[TColSpecN]:
    if not section:
        return []
    entries: list[TColSpecN] = []
    for part in split_om(section):
        if not part:
            continue
        m = PCol.match(part)
        if not m:
            raise SyntaxError
        group_str = m.group("group")
        groups = None if group_str is None else parse_group(group_str)
        entries.append(ColSpec(groups, m.group("name"), m.group("proto")))
    return entries


def runtime_parse(s: str) -> THeadN:
    m = PCol.match(s.strip())
    if not m:
        raise SyntaxError
    group_str = m.group("group")
    groups = parse_group(group_str) if group_str is not None else None
    return Head(
        ColSpec(groups, m.group("name"), m.group("proto")),
        parse_jit(m.group("jit")),
    )

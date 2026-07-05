import awkward as ak
from frozendict import frozendict

try:
    from .colproto import JIT_NAMES, PROTO_NAMES, ColProtoABC
    from .headparser import ColSpec, FrozenGroup, TFrozenColSpec, THeadN, runtime_parse
except ModuleNotFoundError, ImportError:
    from colproto import JIT_NAMES, PROTO_NAMES, ColProtoABC
    from headparser import ColSpec, FrozenGroup, TFrozenColSpec, THeadN, runtime_parse


def load(record_array: ak.Array) -> dict[TFrozenColSpec, ColProtoABC]:
    result: dict[TFrozenColSpec, ColProtoABC] = {}
    fields = record_array.fields

    for col_name in fields:
        head: THeadN = runtime_parse(col_name)

        main_colspecn = head.main

        main_group: FrozenGroup = frozendict(
            {k: (None if v is None else frozenset(v)) for k, v in (main_colspecn.group if main_colspecn.group is not None else {}).items()}
        )

        main_colspec: ColSpec[FrozenGroup] = ColSpec(main_group, main_colspecn.name, main_colspecn.proto)

        main_proto_cls: type[ColProtoABC] | None = PROTO_NAMES.get(main_colspec.proto.upper())
        if main_proto_cls is None:
            raise ValueError(f"未知的协议类型: {main_colspec.proto} (列: {col_name})")

        main_col: ColProtoABC = main_proto_cls(data=record_array[col_name])
        result[main_colspec] = main_col

        for jit_spec_n in head.jit:
            if jit_spec_n.group is None:
                jit_group = main_group
            else:
                jit_group = frozendict(
                    {k: (None if v is None else frozenset(v)) for k, v in (jit_spec_n.group if jit_spec_n.group is not None else {}).items()}
                )

            jit_colspec = ColSpec(jit_group, jit_spec_n.name, jit_spec_n.proto)

            jit_cls = JIT_NAMES.get(jit_colspec.proto.upper())
            if jit_cls is None:
                raise ValueError(f"未知的JIT协议类型: {jit_colspec.proto} (列: {col_name})")

            jit_col = jit_cls(from_=main_col)
            result[jit_colspec] = jit_col

    return result

import re
import types
import typing
from abc import ABC, abstractmethod
from typing import NamedTuple

if __package__:
    from .colproto import ColProtoABC
    from .typing_utils import AwkwardLike
else:
    from colproto import ColProtoABC  # type: ignore
    from typing_utils import AwkwardLike  # type: ignore


schemas = []


def schema(cls: type["SchemaABC"]) -> type["SchemaABC"]:
    schemas.append(cls)
    return cls


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

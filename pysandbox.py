import base64
import pickle
from typing import Any, NamedTuple, Never

import aiohttp
import orjson

TARGET_ADDR = "ka.li"
TARGET_PORT = 59891
BASE_URL = f"http://{TARGET_ADDR}:{TARGET_PORT}"
OCTET_CT = "application/octet-stream"

RETVAL_MISSING = object()


class Err(NamedTuple):
    typ: str | None
    msg: str | None


class Variables(NamedTuple):
    valids: dict[str, Any]
    invalids: dict[str, str]


class Result(NamedTuple):
    retstr: str
    retval: Any
    err: Err
    variables: Variables


def _encode_code(code: str) -> str:
    return base64.a85encode(code.encode("utf-8")).decode("ascii")


def _encode_vars(variables: dict) -> str:
    return base64.a85encode(pickle.dumps(variables)).decode("ascii")


class Sandbox:
    _id: str | None
    _session: aiohttp.ClientSession
    _RETVAL_SENITEL: str

    def __init__(self, *_, **_____) -> Never:
        raise TypeError("不允许同步初始化沙盒；应使用异步的await Sandbox.new()")

    @classmethod
    async def new(cls) -> Sandbox:
        instance = object.__new__(cls)
        instance._id = None
        instance._session = aiohttp.ClientSession()
        try:
            async with instance._session.get(f"{BASE_URL}/pysandbox") as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"创建沙盒失败 (HTTP {resp.status}): {text}")
                data = orjson.loads(await resp.read())
                instance._id = data["id"]
                instance._RETVAL_SENITEL = f"RETVAL_SENITEL_{base64.b16encode(data["id"].encode("ascii")).decode("ascii")}"
        except Exception:
            await instance._session.close()
            raise
        return instance

    async def run(self, code: str, variables: dict) -> Result:
        encoded_code = _encode_code(code)
        encoded_vars = _encode_vars(variables)

        payload = orjson.dumps(
            {
                "id": self._id,
                "code": encoded_code,
                "vars": encoded_vars,
            }
        )

        async with self._session.post(
            f"{BASE_URL}/pysandbox",
            data=payload,
            headers={"Content-Type": OCTET_CT},
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Sandbox execution failed " f"(HTTP {resp.status}): {text}")
            data = orjson.loads(await resp.read())

        err = data.get("err") or {}
        vars_ = data.get("vars") or {}
        retval = data.get("retval")
        return Result(
            retstr=data.get("retstr", ""),
            retval=(RETVAL_MISSING if retval == self._RETVAL_SENITEL else retval),
            err=Err(
                typ=err.get("type"),
                msg=err.get("msg"),
            ),
            variables=Variables(
                valids=vars_.get("valid", {}),
                invalids=vars_.get("invalid", []),
            ),
        )

    async def delete(self) -> None:
        try:
            async with self._session.delete(
                f"{BASE_URL}/pysandbox",
                params={"id": self._id},  # type: ignore
            ) as resp:
                if resp.content_type == OCTET_CT:
                    _ = orjson.loads(await resp.read())  # 这块是面向cv编程的。我也不知道为啥这么写但反正是这么写了那就这么写吧
        finally:
            await self._session.close()

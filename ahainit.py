import base64
import logging
import random
import re
import time
import traceback
import zlib
from collections import deque
from collections.abc import Mapping
from typing import Any, cast

import pinyinparser
import regex

# Aha
from core.api import API
from core.dispatcher import on_cleanup, on_message, on_notice
from fuzzywuzzy import process
from models import api
from models.msg import Forward, MessageChain, MsgChain, Node, Text
from utils.string import InlineStr

from . import lexloader, translator
from .censor import censor
from .translator import BreakOut, SupportsStr, Tianzi

logger = logging.getLogger("tianzi")

last_sent: deque[tuple[str, str]] = deque(maxlen=3)

group_last_command: dict[str, str] = {}

OTHER_BOT_QQ = {
    "1994709738": 1,
    "3491521267": 1,
    "3635837386": 1,  # Lvory
    "3402897586": 2,
    "3498314126": 2,  # 豆鸽
    "3889854671": 3,  # 海狶
    "1825412879": 4,  # 势孙綝
}
OTHER_BOT_RE = {
    re.compile("发 填字[\\s\\S]+"): 1,
    re.compile("(回e)[\\s\\S]+"): 2,
    re.compile(".+"): 3,  # 任何消息都会触发海狶
    re.compile("(说|echo|飞).+"): 4,
}
last_cross_bot_interact: deque[set[int]] = deque(maxlen=36)
last_fuse_time = 0
last_cross_bot_interact_time = 0
passive_call_count = 0
passive_call_hint_count = 5


CRASHACTER = regex.compile(
    "[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f\u1fff\u200b-\u200f\u2028-\u202e\u2060-\u206f\uf900-\ufaff\U000107c0-\U000107ff\U000108b0-\U000108df]"
)

last_tz = Tianzi()


async def tianzi_core(
    raw: str, initial_vars: Mapping[str, SupportsStr] | None = None, initial_inner_vars: Mapping[str, Mapping[str, Any]] | None = None
) -> str:
    try:
        global last_tz
        last_tz = Tianzi(initial_vars=initial_vars, initial_inner_vars=initial_inner_vars)
        translated: str = str(await last_tz.translate(raw, final=True))
    except BreakOut as bo:
        translated = bo.args[0]
    except TimeoutError:
        translated = "正则超时。您的输入可能太长或太复杂。"
    except Exception as e:
        translated = f"意外的错误：{repr(e)}，栈如下：\n{traceback.format_exc()}\n请联系找北。"
    if last_tz.current_stat.censor:
        translated = censor(translated)
    ret_msg = regex.sub(CRASHACTER, lambda char: f"U+{hex(ord(char.group(0)))[2:].upper()}", translated.removeprefix("\n").removeprefix(" "))
    logger.info(f"CALL STACK: {"".join(last_tz.call_stack)}")

    return ret_msg


def fused(event: api.Message, ret: str) -> str | None:
    triggered_by = OTHER_BOT_QQ.get(event.user_id)
    triggers = next((v for (k, v) in OTHER_BOT_RE.items() if k.fullmatch(ret)), None)
    global last_fuse_time, last_cross_bot_interact_time, passive_call_count, passive_call_hint_count
    if triggered_by:
        passive_call_count += 1
        if passive_call_count >= 16:
            last_fuse_time = time.monotonic()
            if passive_call_hint_count > 0:
                passive_call_hint_count -= 1
                return f"【开环熔断】Bot被交互深度超出限制。本次本消息{f"最多还会再显示{passive_call_hint_count}次。" if passive_call_hint_count > 0 else "将不再显示。"}"
                # 因为本bot没有管理员的话不能主动打破开环，所以只能避免本机反复发提示
        if (curr_t := time.monotonic()) - last_fuse_time < 15:
            return "【熔断冷却】Bot互交互深度超限后的15秒内不可再使用其他bot调用薨机。"
    if triggered_by and triggers:
        last_cross_bot_interact_time = curr_t
        last_cross_bot_interact.append({triggered_by, triggers})
        if last_cross_bot_interact.count(last_cross_bot_interact[-1]) >= 5:
            last_fuse_time = curr_t
            return "【闭环熔断】Bot互交互深度超出限制。"
    elif time.monotonic() - last_cross_bot_interact_time > 15:
        last_cross_bot_interact.clear()
        passive_call_count = 0
        passive_call_hint_count = 5


@on_message("不发 填字(-V)?[\\s\n][\\s\\S]+")
async def tianzi(event: api.Message):
    raw = event.get_msg_inline().removeprefix("不发 填字")
    if raw.startswith("-V"):
        raw = re.sub(
            "\n *(?!\n)",
            "",
            raw[2:],
        )
    raw = raw.removeprefix(" ")

    uid = event.user_id
    gid = event.group_id

    ret_msg = InlineStr(await tianzi_core(raw, initial_inner_vars={"QQAPI": {"uid": uid, "gid": gid or ""}}))

    if fusemsg := fused(event, ret_msg):
        await event.send(fusemsg)
        return

    if ret_msg.startswith("名 "):
        await event.send("名 薨机")
        return
    if ret_msg.startswith("发 我叫"):
        await event.send("发 我叫 薨机")
        return

    if last_tz.current_stat.pua_warning:
        await event.send(
            "警告：输入/处理过程中涉及保留的PUA字符，可能导致意外的行为。请不要使用PUA字符，或者如果您明白自己在做什么的话，请使用[[config<disable_pua_warning>]]消除本警告。如果您认为这是误报，请@找北。"
        )
    if ret_msg:
        mid = await event.send(ret_msg.to_list())
        last_sent.append((str(event.message_id), mid))
        group_last_command[str(event.group_id)] = raw
        if ret_msg.startswith("不发 填字") and ret_msg != "不发 填字 自己吓自己~" and ret_msg != "不发 填字 自己吓自己～":
            await event.send("自己吓自己~")


@on_message("不发 填字同上([\\s][\\s\\S]+)?")
async def tianzi_repeat(event: api.Message):
    # 我知道这坨玩意冗余太多了但是我懒得改反正能用就行我管你这个那个的（不是

    variants_str = str(event.get_msg_inline())[7:].strip().split("\n")

    uid = event.user_id
    gid = event.group_id

    try:
        variants = {(sp := ln.split("=", maxsplit=1))[0]: sp[1] for ln in variants_str if ln}
    except IndexError:
        await event.send("填字同上指定变量格式错误：应当使用每行一个的「变量名=值」格式。")
        return
    if str(event.group_id) in group_last_command:
        ret_msg = await tianzi_core(
            group_last_command[str(event.group_id)], variants, initial_inner_vars={"QQAPI": {"uid": uid, "gid": gid or ""}}
        )

        if fusemsg := fused(event, ret_msg):
            await event.send(fusemsg)
            return

        if ret_msg.startswith("名 "):
            await event.send("名 薨机")
            return
        if ret_msg.startswith("发 我叫"):
            await event.send("发 我叫 薨机")
            return
        if ret_msg:
            mid = await event.send(ret_msg)
            last_sent.append((str(event.message_id), mid))


@on_message("不发 填字同上是(啥|什么)")
async def tianzi_repeat_what(event: api.Message):
    if str(event.group_id) in group_last_command:
        await event.send(group_last_command[str(event.group_id)])
    else:
        await event.send("上次重启之后本群还没有进行过填字。")


@on_message("不发 填字调用栈")
async def tianzi_callstack(event: api.Message):
    await event.send("".join(last_tz.call_stack))


@on_message("不发 帮助 填字")
async def help_tianzi(event: api.Message):
    await event.send(translator.helps())


@on_message("不发 工资")
async def linecount(event: api.Message):
    from .. import linecount

    lc = linecount.count_python_lines()
    await event.send(f"你要想不发这个，你得先给薨机不发工资，薨机有{lc}行代码，一行一块钱")


@on_notice("group_recall")
async def del_tianzi(event: api.Notice):
    rid = event.message_id
    for recv_mid, sent_mid in last_sent:
        if recv_mid == rid:
            await API.delete_msg(message_id=sent_mid)
            break


@on_message("不发 查词[库典].*")
async def get_lex_meta(event: api.Message):
    if name := event.message_str[6:].lstrip():
        try:
            meta = await (await lexloader.Lexicon.load(name)).fmt_metadata()
        except KeyError:
            fuzz = cast(list[tuple[str, int]], process.extract(name, lexloader.all_lexicons, limit=2))
            if fuzz and fuzz[0][1] >= 70:
                if len(fuzz) == 2 and fuzz[1][1] == fuzz[0][1]:
                    hint = f"猜你想找：「{fuzz[0][0]}」「{fuzz[1][0]}」"
                else:
                    hint = f"猜你想找：「{fuzz[0][0]}」"
            else:
                hint = ""
            meta = f"词库不存在。{hint}"
        await event.send(meta)
    else:
        names = lexloader.all_lexicons
        namesegs = [
            Node(nickname=f"词典{i+1}~{i+50}", content=MessageChain(Text(text="\n".join(names[i : i + 50])))) for i in range(0, len(names), 50)
        ]
        await event.send(Forward(content=MsgChain(namesegs)))


"""@on_message("不发 查词 .+")
async def find_lex(event: api.Message):
    word = event.message_str[6:].strip()
    lexs = await lexloaders.LexLoader.search_word(word)
    if not lexs:
        ret = f"{word} - 不见于任何词典"
    else:
        lex_names = [lex.meta.name for lex in lexs]
        ret = f"{word} - 见于以下{len(lexs)}个词典：「{"」「".join(lex_names)}」"
    await event.send(ret)"""


@on_message("不发 拼解 .+")
async def pinyin_parse(event: api.Message):
    pinyin = event.message_str[6:].strip()
    try:
        ss = pinyinparser.parse(pinyin)
        await event.send(f"{" ".join(repr(s) for s in ss)} → {pinyinparser.syllables_to_str(ss)}")
    except ValueError:
        await event.send("无法解析")


GY_CHARSET = "古咕孤故谷估菇固顾姑蛄鼓沽诂牯菰"
GY_ORIGINAL = ["2D31FC5E97684B0A", "0ACD7B29836E4F51", "3A26075EFC4B81D9", "392AC5780F14B6ED", "A4DE3B586027F1C9"]


@on_message("不发 鸽曰\\+ .+")
async def geyue_cipher(event: api.Message):
    msg = event.message_str[6:].strip()
    orig_used = random.randint(0, 4)
    trans = str.maketrans(GY_ORIGINAL[orig_used], GY_CHARSET)
    data = zlib.compress(msg.encode("utf8"), 9)
    if len(data) > 768:
        await event.send("让我咕这么长一坨你是要累死我吗😡👊")
        return
    ciphered = base64.b16encode(data).decode("ascii").translate(trans)
    pad = 3 - (len(ciphered)) % 3  # 用最后一个字符标识pad了多长以及使用的顺序，因此pad长度是1/2/3而不是0/1/2（也就是一定会有pad）
    padchar = GY_CHARSET[pad - 1 + orig_used * 3]
    ciphered += "".join(random.choices(GY_CHARSET, k=pad - 1)) + padchar
    await event.send("！".join((ciphered[i : i + 3] for i in range(0, len(ciphered), 3))) + "！")


@on_message("不发 鸽曰\\- .+")
async def geyue_decipher(event: api.Message):
    try:
        msg = event.message_str[6:].strip().replace("！", "")
        padchar = msg[-1]
        padinfo = GY_CHARSET.index(padchar)
        pad, orig_used = padinfo % 3 + 1, padinfo // 3
        msg = msg[:-pad]
        trans = str.maketrans(GY_CHARSET, GY_ORIGINAL[orig_used])
        deciphered_data = base64.b16decode(msg.translate(trans))
        deciphered = zlib.decompress(deciphered_data).decode("utf8")
        if len(deciphered) / len(msg) > 8:
            await event.send("压缩炸弹是吧😡👊")
            return
        await event.send(deciphered)
    except Exception:
        await event.send("咕咕嘎嘎的说什么呢，听不懂")


@on_cleanup()
async def finalize():
    if last_tz._sandbox:
        await last_tz._sandbox.delete()

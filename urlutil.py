import httpx


async def shorturl(url: str, timeout: int = 1) -> str:
    try:
        async with httpx.AsyncClient() as cl:
            ret = await cl.post(
                "http://127.0.0.1:59977/internal/create",
                headers={"Content-Type": "application/json"},
                json={"url": url},
                timeout=timeout,
            )
    except httpx.TimeoutException:
        return url
    if ret.status_code != 200:
        return url
    json = ret.json()
    return f"https://hongbot.icu/l/{json["id"]}"


# 自己实际用的时候换成urlvanish或者其他别的短链接服务就行，懒的话直接return url都完全没问题

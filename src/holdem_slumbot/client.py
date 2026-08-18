"""Slumbot HTTP 客户端——**全项目唯一联网的地方**。

只有两个接口：`new_hand` 开一手、`act` 走一步。协议细节写在 `protocol.py` 的模块说明里。
把网络这一层单独关在这里，是为了让协议翻译与对局统计都能脱离网络单测。

```python
session = Session()
body = session.new_hand()
body = session.act("b200")
```
"""

from __future__ import annotations

import json
import time
import urllib.request

__all__ = ["BASE", "SlumbotError", "Session"]

BASE = "https://slumbot.com/api"


class SlumbotError(RuntimeError):
    """Slumbot 判我们的动作非法，或它那边出了状况。"""


class Session:
    """一条会话：`token` 把连续的手牌串起来，位置逐手轮换。

    出错之后把 `token` 置空即可换一条新会话重开——那一手作废，别把它的观察算进统计。
    """

    def __init__(self, timeout: float = 20.0, pause: float = 0.05) -> None:
        self.token: str | None = None
        self.timeout = timeout
        self.pause = pause
        """每次请求后歇一下，别把免费服务打疼了。"""
        self.requests = 0

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{BASE}/{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read())
        self.requests += 1
        if "error_msg" in body:
            # 字段名是 error_msg，不是 error_message——实测所得
            raise SlumbotError(f"{body['error_msg']}（此前动作 {body.get('old_action')!r}）")
        if "token" in body:
            self.token = body["token"]
        if self.pause:
            time.sleep(self.pause)
        return body

    def new_hand(self) -> dict:
        return self._post("new_hand", {"token": self.token} if self.token else {})

    def act(self, incr: str) -> dict:
        return self._post("act", {"token": self.token, "incr": incr})

    def reset(self) -> None:
        """丢掉会话，下一手从头开始。"""
        self.token = None

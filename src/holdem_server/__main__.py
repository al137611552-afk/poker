"""启动本地服务端：

    python -m holdem_server                # 默认 0.0.0.0:8000
    python -m holdem_server --port 9000

默认绑定 0.0.0.0 是刻意的——手机要在同一 Wi-Fi 下通过电脑的局域网 IP 访问。
这也意味着**同一网络内的其他设备都能打开它**，公共网络下请改用 --host 127.0.0.1。
"""

from __future__ import annotations

import argparse
import socket

import uvicorn


def local_ip() -> str:
    """取本机在局域网中的地址，只为打印一个手机能直接输入的网址。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.255.255.255", 1))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="德扑训练台 · 本地服务端")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", default="hands.sqlite", help="牌谱数据库路径")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.host == "0.0.0.0":
        print(f"电脑上打开： http://127.0.0.1:{args.port}")
        print(f"手机上打开： http://{local_ip()}:{args.port}  （需在同一 Wi-Fi）")
    print(f"牌谱数据库： {args.db}\n")

    import os

    os.environ.setdefault("HOLDEM_DB", args.db)
    uvicorn.run(
        "holdem_server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

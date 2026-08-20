# -*- coding: utf-8 -*-
"""Helloworld HTTP 接口层（Controller 层）与服务启动入口。

基于 Python 标准库 http.server 实现，零第三方依赖。
接口协议：GET /api/helloworld
运行方式：python helloworld_app.py [--port 8000]
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from helloworld_service import HelloworldService

# 常量定义（禁止魔法值）：接口路径与端口默认值
API_PATH_HELLOWORLD = "/api/helloworld"
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"
ENV_KEY_PORT = "PORT"

# 常量定义：响应码与提示文案
CODE_SUCCESS = "00000"
CODE_USER_ERROR = "A0001"
MESSAGE_SUCCESS = "success"
MESSAGE_METHOD_NOT_ALLOWED = "method not allowed: only GET is supported"
MESSAGE_PATH_NOT_FOUND = "resource not found"
USER_TIP_SUCCESS = ""
USER_TIP_METHOD_NOT_ALLOWED = "仅支持 GET 请求"
USER_TIP_PATH_NOT_FOUND = "请求的资源不存在"
USER_TIP_HELLO_WORLD = "Hello, World!"


class HelloworldHttpHandler(BaseHTTPRequestHandler):
    """Helloworld 接口请求处理器。

    负责 HTTP 协议处理、路径与请求方法分发、统一响应结构组装：
    所有响应统一使用 lowerCamelCase 键名，错误响应包含
    errorCode / errorMessage / userTip / data 四个部分。
    """

    # 业务服务（无状态单例，进程内共享）
    service = HelloworldService()

    def handle_request(self):
        """分发并处理请求（不依赖真实 socket，便于单元测试）。

        :return: (http_status, payload) 二元组，payload 为统一响应结构
        """
        if self.path != API_PATH_HELLOWORLD:
            return 404, self._build_payload(
                CODE_USER_ERROR, MESSAGE_PATH_NOT_FOUND, USER_TIP_PATH_NOT_FOUND
            )
        if self.command != "GET":
            return 405, self._build_payload(
                CODE_USER_ERROR, MESSAGE_METHOD_NOT_ALLOWED, USER_TIP_METHOD_NOT_ALLOWED
            )
        greeting = self.service.get_greeting()
        return 200, self._build_payload(
            CODE_SUCCESS,
            MESSAGE_SUCCESS,
            USER_TIP_SUCCESS,
            data={"greetMsg": greeting},
        )

    def do_GET(self):
        """处理 GET 请求。"""
        self._send_response(*self.handle_request())

    def do_HEAD(self):
        """处理 HEAD 请求（统一返回 405，避免落入默认 501）。"""
        self._send_response(*self.handle_request())

    def do_POST(self):
        """处理 POST 请求（统一返回 405）。"""
        self._send_response(*self.handle_request())

    def do_PUT(self):
        """处理 PUT 请求（统一返回 405）。"""
        self._send_response(*self.handle_request())

    def do_DELETE(self):
        """处理 DELETE 请求（统一返回 405）。"""
        self._send_response(*self.handle_request())

    def _send_response(self, status_code, payload):
        """按统一结构写出响应。

        :param status_code: HTTP 状态码
        :param payload: 响应体字典
        """
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _build_payload(error_code, error_message, user_tip, data=None):
        """组装统一响应结构。

        :param error_code: 错误码（00000 表示成功）
        :param error_message: 可排查的错误信息
        :param user_tip: 面向用户的提示信息
        :param data: 业务数据，默认为 None
        :return: 响应体字典
        """
        return {
            "errorCode": error_code,
            "errorMessage": error_message,
            "userTip": user_tip,
            "data": data,
        }

    def log_message(self, format_string, *args):
        """静默访问日志，避免向 stderr 输出干扰。"""
        return


def _parse_port(arguments):
    """从命令行参数中解析 --port 的取值。

    :param arguments: 命令行参数列表（不含程序名）
    :return: 端口值，未指定时返回 None
    """
    if "--port" in arguments:
        index = arguments.index("--port")
        if index + 1 < len(arguments):
            return int(arguments[index + 1])
    return None


def _resolve_port():
    """解析服务监听端口：优先级为命令行参数 > 环境变量 > 默认值。

    :return: 监听端口
    """
    cli_port = _parse_port(sys.argv[1:])
    if cli_port is not None:
        return cli_port
    env_port = os.getenv(ENV_KEY_PORT)
    if env_port is not None:
        return int(env_port)
    return DEFAULT_PORT


def create_server(port=DEFAULT_PORT):
    """创建 Helloworld HTTP 服务。

    :param port: 监听端口
    :return: HTTPServer 实例
    """
    return HTTPServer((DEFAULT_HOST, port), HelloworldHttpHandler)


def main():
    """服务启动入口：python helloworld_app.py [--port 8000]。"""
    port = _resolve_port()
    server = create_server(port)
    print("HelloWorld service started at http://%s:%d/api/helloworld" % (DEFAULT_HOST, port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nHelloWorld service stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
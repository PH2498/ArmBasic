# -*- coding: utf-8 -*-
"""Helloworld 接口模块单元测试。

遵循 FIRST 原则（Fast, Independent, Repeatable, Self-validating, Timely），
基于 Python 标准库 unittest 编写，无第三方依赖。
"""
import json
import unittest

from helloworld_app import HelloworldHttpHandler
from helloworld_service import HelloworldService


def _build_handler(method, path):
    """构造 HelloworldHttpHandler 测试实例。

    仅用于验证请求分发与响应组装逻辑，不发起真实网络请求，
    保证测试快速且可重复执行。

    :param method: HTTP 请求方法，如 GET/POST
    :param path: 请求路径，如 /api/helloworld
    :return: 未绑定 socket 的 handler 实例
    """
    handler = HelloworldHttpHandler.__new__(HelloworldHttpHandler)
    handler.command = method
    handler.path = path
    return handler


class HelloworldServiceTest(unittest.TestCase):
    """HelloworldService 业务层测试。"""

    def test_should_return_greeting_when_call_get_greeting(self):
        """正常路径：调用 get_greeting 应返回 Hello, World!"""
        service = HelloworldService()

        greeting = service.get_greeting()

        self.assertEqual("Hello, World!", greeting)


class HelloworldHttpHandlerTest(unittest.TestCase):
    """HelloworldHttpHandler 接口层测试。"""

    def test_should_return_success_when_get_helloworld(self):
        """正常路径：GET /api/helloworld 应返回成功响应与问候语。"""
        handler = _build_handler("GET", "/api/helloworld")

        status_code, payload = handler.handle_request()

        self.assertEqual(200, status_code)
        self.assertEqual("00000", payload["errorCode"])
        self.assertEqual("success", payload["errorMessage"])
        self.assertEqual("Hello, World!", payload["data"]["greetMsg"])

    def test_should_use_lower_camel_case_for_json_keys(self):
        """JSON 键命名：响应体所有 key 必须为 lowerCamelCase。"""
        handler = _build_handler("GET", "/api/helloworld")

        _status_code, payload = handler.handle_request()

        raw_json = json.dumps(payload)

        self.assertIn('"errorCode"', raw_json)
        self.assertIn('"errorMessage"', raw_json)
        self.assertIn('"userTip"', raw_json)
        self.assertIn('"greetMsg"', raw_json)
        self.assertNotIn("error_code", raw_json)

    def test_should_return_405_when_method_not_allowed(self):
        """异常路径：非 GET 方法应返回 405 与用户级错误码。"""
        for method in ("POST", "PUT", "DELETE", "HEAD"):
            handler = _build_handler(method, "/api/helloworld")

            status_code, payload = handler.handle_request()

            self.assertEqual(405, status_code, msg="method=%s" % method)
            self.assertEqual("A0001", payload["errorCode"], msg="method=%s" % method)
            self.assertIsNone(payload["data"], msg="method=%s" % method)

    def test_should_return_404_when_path_not_found(self):
        """边界值：GET 未知路径应返回 404 与用户级错误码。"""
        handler = _build_handler("GET", "/api/unknown")

        status_code, payload = handler.handle_request()

        self.assertEqual(404, status_code)
        self.assertEqual("A0001", payload["errorCode"])
        self.assertIsNone(payload["data"])


if __name__ == "__main__":
    unittest.main()
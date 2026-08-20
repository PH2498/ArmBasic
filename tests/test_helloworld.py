"""
HelloWorld API 单元测试。

测试范围：
- HelloService.get_greeting() 业务逻辑
- Flask 路由层请求/响应
"""

import sys
import os
import pytest

# 将项目根目录加入路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from HelloWorldApi.service import HelloService
from HelloWorldApi.app import create_app


class TestHelloService:
    """HelloService 业务逻辑单元测试"""

    def setup_method(self):
        self.service = HelloService()

    # ── 正常路径 ──────────────────────────────────────

    def test_should_return_default_greeting_when_name_is_none(self):
        """无参数时返回默认问候语"""
        result = self.service.get_greeting()
        assert result == "Hello, World!"

    def test_should_return_personalized_greeting_when_name_provided(self):
        """传入 name 参数时返回个性化问候"""
        result = self.service.get_greeting("Alice")
        assert result == "Hello, Alice!"

    def test_should_return_personalized_greeting_when_name_is_chinese(self):
        """传入中文名称"""
        result = self.service.get_greeting("小明")
        assert result == "Hello, 小明!"

    # ── 边界值 ──────────────────────────────────────

    def test_should_use_default_when_name_is_empty_string(self):
        """name 为空字符串时使用默认值 'World'"""
        result = self.service.get_greeting("")
        assert result == "Hello, World!"

    def test_should_use_default_when_name_is_whitespace_only(self):
        """name 为纯空白字符时使用默认值（strip 后为空）"""
        result = self.service.get_greeting("   ")
        assert result == "Hello, World!"

    def test_should_truncate_when_name_exceeds_100_chars(self):
        """name 长度超过 100 字符时截断"""
        long_name = "A" * 150
        result = self.service.get_greeting(long_name)
        expected = "Hello, " + ("A" * 100) + "!"
        assert result == expected
        assert len(result) == 108  # "Hello, " (7) + 100 chars + "!" (1) = 108

    def test_should_handle_exactly_100_chars(self):
        """name 恰好 100 字符，不截断"""
        name = "A" * 100
        result = self.service.get_greeting(name)
        expected = "Hello, " + name + "!"
        assert result == expected

    # ── 特殊字符 ──────────────────────────────────────

    def test_should_handle_special_characters_in_name(self):
        """name 包含特殊字符，直接拼接不做过滤"""
        result = self.service.get_greeting("<script>alert(1)</script>")
        assert result == "Hello, <script>alert(1)</script>!"

    def test_should_handle_numbers_in_name(self):
        """name 包含数字"""
        result = self.service.get_greeting("User123")
        assert result == "Hello, User123!"


class TestHelloWorldRoute:
    """Flask 路由层集成测试"""

    @pytest.fixture
    def client(self):
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    # ── API-01: 无参数 helloworld ─────────────────────

    def test_should_return_default_greeting_on_get(self, client):
        """GET /api/helloworld 无参数，返回默认问候"""
        response = client.get("/api/helloworld")
        assert response.status_code == 200
        data = response.get_json()
        assert data["code"] == 0
        assert data["msg"] == "success"
        assert data["data"]["greeting"] == "Hello, World!"

    # ── API-02: 参数化问候 ───────────────────────────

    def test_should_return_personalized_greeting_with_name_param(self, client):
        """GET /api/helloworld?name=Alice 返回个性化问候"""
        response = client.get("/api/helloworld?name=Alice")
        assert response.status_code == 200
        data = response.get_json()
        assert data["code"] == 0
        assert data["data"]["greeting"] == "Hello, Alice!"

    def test_should_use_default_when_name_param_is_empty(self, client):
        """GET /api/helloworld?name= 空参数，使用默认值"""
        response = client.get("/api/helloworld?name=")
        assert response.status_code == 200
        data = response.get_json()
        assert data["data"]["greeting"] == "Hello, World!"

    # ── HTTP 方法错误 ────────────────────────────────

    def test_should_return_405_on_post(self, client):
        """POST /api/helloworld 应返回 405 Method Not Allowed"""
        response = client.post("/api/helloworld")
        assert response.status_code == 405

    # ── 响应格式校验 ─────────────────────────────────

    def test_should_return_correct_json_structure(self, client):
        """响应 JSON 结构应包含 code, msg, data 字段"""
        response = client.get("/api/helloworld")
        data = response.get_json()
        assert "code" in data
        assert "msg" in data
        assert "data" in data
        assert "greeting" in data["data"]
        assert isinstance(data["code"], int)
        assert isinstance(data["msg"], str)
        assert isinstance(data["data"]["greeting"], str)

    def test_should_return_content_type_json(self, client):
        """响应 Content-Type 应为 application/json"""
        response = client.get("/api/helloworld")
        assert response.content_type == "application/json"
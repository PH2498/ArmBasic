"""
HelloWorld 路由层 - Flask Blueprint 注册。

对外暴露 RESTful API 端点。
"""

from flask import Blueprint, request, jsonify

from HelloWorldApi.service import HelloService

hello_bp = Blueprint("helloworld", __name__, url_prefix="/api")

_service = HelloService()


@hello_bp.route("/helloworld", methods=["GET"])
def helloworld():
    """helloworld 问候接口。

    GET /api/helloworld         → 默认问候 "Hello, World!"
    GET /api/helloworld?name=X  → 个性化问候 "Hello, X!"
    """
    name = request.args.get("name", default=None)
    greeting = _service.get_greeting(name)

    return jsonify(
        {
            "code": 0,
            "msg": "success",
            "data": {"greeting": greeting},
        }
    )
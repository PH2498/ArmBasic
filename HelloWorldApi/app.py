"""
HelloWorldApi Flask 应用入口。

初始化 Flask 应用、注册 Blueprint、加载配置。
"""

import os
from flask import Flask


def create_app(config: dict | None = None) -> Flask:
    """创建并配置 Flask 应用。

    Args:
        config: 可选的配置字典，用于测试环境覆盖。

    Returns:
        配置完成的 Flask 应用实例。
    """
    app = Flask(__name__)

    # 默认配置
    app.config.setdefault("JSONIFY_PRETTYPRINT_REGULAR", False)

    if config:
        app.config.update(config)

    # 注册 Blueprint
    from HelloWorldApi.routes import hello_bp

    app.register_blueprint(hello_bp)

    return app


def main():
    """启动入口：创建应用并运行。"""
    app = create_app()
    port = int(os.environ.get("HELLOWORLD_PORT", 5000))
    debug = os.environ.get("HELLOWORLD_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""Helloworld 业务服务层。

对应分层架构中的 Service 层：封装问候语生成等业务逻辑，
不依赖任何 HTTP 协议细节，便于单元测试。
"""


class HelloworldService:
    """Helloworld 业务服务，负责生成问候语。"""

    # 常量定义（禁止魔法值）：默认问候语文本
    DEFAULT_GREETING = "Hello, World!"

    def get_greeting(self):
        """获取默认问候语。

        :return: 问候语文本
        """
        return self.DEFAULT_GREETING
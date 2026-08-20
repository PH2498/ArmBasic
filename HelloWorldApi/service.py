"""
HelloService - HelloWorld 业务逻辑层。

纯函数，无状态，负责问候语拼装与参数校验。
"""

from typing import ClassVar


class HelloService:
    """问候服务：根据名称参数生成问候语。"""

    DEFAULT_NAME: ClassVar[str] = "World"
    MAX_NAME_LENGTH: ClassVar[int] = 100

    def get_greeting(self, name: str | None = None) -> str:
        """生成问候语。

        Args:
            name: 问候对象名称，可选。为 None 或空字符串时使用默认值 "World"；
                  长度超过 100 字符时自动截断。

        Returns:
            格式为 "Hello, {name}!" 的问候语字符串。
        """
        if name is None:
            name = self.DEFAULT_NAME
        else:
            name = name.strip()
            if not name:
                name = self.DEFAULT_NAME
            elif len(name) > self.MAX_NAME_LENGTH:
                name = name[: self.MAX_NAME_LENGTH]

        return f"Hello, {name}!"
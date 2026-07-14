"""
输入验证器
实现用户输入数据的验证功能
"""
import re
from typing import Tuple, Optional


class InputValidator:
    """输入验证器"""
    
    # 用户名规则：4-50字符，字母数字下划线
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_]{4,50}$')
    
    # 密码规则：至少8位，包含大小写字母和数字
    PASSWORD_PATTERN = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$')
    
    @staticmethod
    def validate_username(username: str) -> Tuple[bool, Optional[str]]:
        """验证用户名
        
        Args:
            username: 用户名
            
        Returns:
            (是否有效, 错误消息)
        """
        if not username:
            return False, "用户名不能为空"
        
        if not isinstance(username, str):
            return False, "用户名必须是字符串"
        
        username = username.strip()
        
        if len(username) < 4:
            return False, "用户名长度不能少于4个字符"
        
        if len(username) > 50:
            return False, "用户名长度不能超过50个字符"
        
        if not InputValidator.USERNAME_PATTERN.match(username):
            return False, "用户名只能包含字母、数字和下划线"
        
        return True, None
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, Optional[str]]:
        """验证密码强度
        
        Args:
            password: 密码
            
        Returns:
            (是否有效, 错误消息)
        """
        if not password:
            return False, "密码不能为空"
        
        if not isinstance(password, str):
            return False, "密码必须是字符串"
        
        if len(password) < 8:
            return False, "密码长度不能少于8个字符"
        
        if len(password) > 128:
            return False, "密码长度不能超过128个字符"
        
        if not InputValidator.PASSWORD_PATTERN.match(password):
            return False, "密码必须包含大小写字母和数字"
        
        return True, None
    
    @staticmethod
    def validate_login_input(username: str, password: str) -> Tuple[bool, Optional[str]]:
        """验证登录输入
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            (是否有效, 错误消息)
        """
        # 用户名验证（登录时允许更宽松的格式）
        if not username or not isinstance(username, str):
            return False, "用户名格式错误"
        
        username = username.strip()
        if len(username) < 1 or len(username) > 50:
            return False, "用户名长度错误"
        
        # 密码验证
        if not password or not isinstance(password, str):
            return False, "密码格式错误"
        
        if len(password) < 1:
            return False, "密码不能为空"
        
        return True, None
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """清理输入文本（防止 XSS）
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 移除危险的 HTML 标签
        text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<.*?>', '', text)
        
        return text.strip()
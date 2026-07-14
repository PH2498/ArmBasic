"""
JWT Token 处理工具
实现 JWT Token 的生成、验证和刷新功能
"""
import jwt
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from ..config import Config


class JWTHandler:
    """JWT Token 处理器"""
    
    ALGORITHM = 'HS256'
    ACCESS_TOKEN_EXPIRES = 24 * 60 * 60  # 24小时（秒）
    REFRESH_TOKEN_EXPIRES = 7 * 24 * 60 * 60  # 7天（秒）
    
    @staticmethod
    def generate_access_token(user_id: int, username: str) -> str:
        """生成访问令牌
        
        Args:
            user_id: 用户ID
            username: 用户名
            
        Returns:
            JWT Token 字符串
        """
        now = datetime.utcnow()
        jti = str(uuid.uuid4())
        
        payload = {
            'user_id': user_id,
            'username': username,
            'type': 'access',
            'jti': jti,
            'iat': now,
            'exp': now + timedelta(seconds=JWTHandler.ACCESS_TOKEN_EXPIRES)
        }
        
        return jwt.encode(payload, Config.SECRET_KEY, algorithm=JWTHandler.ALGORITHM)
    
    @staticmethod
    def generate_refresh_token(user_id: int) -> str:
        """生成刷新令牌
        
        Args:
            user_id: 用户ID
            
        Returns:
            Refresh Token 字符串
        """
        now = datetime.utcnow()
        jti = str(uuid.uuid4())
        
        payload = {
            'user_id': user_id,
            'type': 'refresh',
            'jti': jti,
            'iat': now,
            'exp': now + timedelta(seconds=JWTHandler.REFRESH_TOKEN_EXPIRES)
        }
        
        return jwt.encode(payload, Config.SECRET_KEY, algorithm=JWTHandler.ALGORITHM)
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """验证 Token 并返回载荷
        
        Args:
            token: JWT Token 字符串
            
        Returns:
            解码后的载荷字典，验证失败返回 None
        """
        try:
            payload = jwt.decode(
                token, 
                Config.SECRET_KEY, 
                algorithms=[JWTHandler.ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    @staticmethod
    def get_jti(token: str) -> Optional[str]:
        """获取 Token 的唯一标识（JTI）
        
        Args:
            token: JWT Token 字符串
            
        Returns:
            JTI 字符串，解析失败返回 None
        """
        payload = JWTHandler.verify_token(token)
        return payload.get('jti') if payload else None
    
    @staticmethod
    def get_user_id(token: str) -> Optional[int]:
        """从 Token 中提取用户ID
        
        Args:
            token: JWT Token 字符串
            
        Returns:
            用户ID，解析失败返回 None
        """
        payload = JWTHandler.verify_token(token)
        return payload.get('user_id') if payload else None
    
    @staticmethod
    def refresh_access_token(refresh_token: str) -> Optional[str]:
        """使用刷新令牌生成新的访问令牌
        
        Args:
            refresh_token: 刷新令牌
            
        Returns:
            新的访问令牌，失败返回 None
        """
        payload = JWTHandler.verify_token(refresh_token)
        
        if not payload:
            return None
        
        if payload.get('type') != 'refresh':
            return None
        
        user_id = payload.get('user_id')
        if not user_id:
            return None
        
        # 这里需要从数据库获取用户名，暂时返回 None
        # 实际使用时应该在 auth_service 中调用
        return None
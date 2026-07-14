"""
认证服务核心
实现用户注册、登录、登出等核心认证功能
"""
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from sqlalchemy.exc import IntegrityError
from .user_model import User, TokenBlacklist, LoginAttempt, get_session
from .utils.jwt_handler import JWTHandler
from .utils.validators import InputValidator
from .config import Config


class AuthService:
    """认证服务类"""
    
    def __init__(self):
        """初始化认证服务 - 登录失败计数通过数据库持久化"""
        pass
    
    def register_user(self, username: str, password: str) -> Tuple[bool, str, Optional[Dict]]:
        """用户注册
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            (是否成功, 消息, 用户数据)
        """
        # 验证输入
        valid, msg = InputValidator.validate_username(username)
        if not valid:
            return False, msg, None
        
        valid, msg = InputValidator.validate_password(password)
        if not valid:
            return False, msg, None
        
        # 清理用户名
        username = InputValidator.sanitize_input(username)
        
        # 生成密码哈希
        password_hash = self._hash_password(password)
        
        # 创建用户
        session = get_session()
        try:
            user = User(
                username=username,
                password_hash=password_hash,
                is_active=True
            )
            session.add(user)
            session.commit()
            
            return True, "注册成功", {
                'user_id': user.id,
                'username': user.username
            }
        except IntegrityError:
            session.rollback()
            return False, "用户名已存在", None
        except Exception as e:
            session.rollback()
            return False, "注册失败，请稍后重试", None
        finally:
            session.close()
    
    def login(self, username: str, password: str) -> Tuple[bool, str, Optional[Dict]]:
        """用户登录
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            (是否成功, 消息, 认证数据)
        """
        # 验证输入
        valid, msg = InputValidator.validate_login_input(username, password)
        if not valid:
            return False, msg, None
        
        username = InputValidator.sanitize_input(username)
        
        # 检查登录锁定
        if self._is_locked_out(username):
            return False, "账户已被锁定，请15分钟后再试", None
        
        # 查询用户
        session = get_session()
        try:
            user = session.query(User).filter(
                User.username == username,
                User.is_active == True
            ).first()
            
            if not user:
                self._record_login_attempt(username)
                return False, "用户名或密码错误", None
            
            # 验证密码
            if not self._verify_password(password, user.password_hash):
                self._record_login_attempt(username)
                return False, "用户名或密码错误", None
            
            # 登录成功，清除登录失败记录
            self._clear_login_attempts(username)
            
            # 生成 Token
            access_token = JWTHandler.generate_access_token(user.id, user.username)
            refresh_token = JWTHandler.generate_refresh_token(user.id)
            
            return True, "登录成功", {
                'user_id': user.id,
                'username': user.username,
                'access_token': access_token,
                'refresh_token': refresh_token,
                'expires_in': JWTHandler.ACCESS_TOKEN_EXPIRES
            }
        except Exception as e:
            session.rollback()
            return False, "登录失败，请稍后重试", None
        finally:
            session.close()
    
    def logout(self, token: str) -> Tuple[bool, str]:
        """用户登出（将 Token 加入黑名单）
        
        Args:
            token: JWT Token
            
        Returns:
            (是否成功, 消息)
        """
        jti = JWTHandler.get_jti(token)
        if not jti:
            return False, "无效的 Token"
        
        session = get_session()
        try:
            # 检查是否已在黑名单
            existing = session.query(TokenBlacklist).filter(
                TokenBlacklist.token_jti == jti
            ).first()
            
            if existing:
                return True, "已登出"
            
            # 添加到黑名单
            blacklist_entry = TokenBlacklist(token_jti=jti)
            session.add(blacklist_entry)
            session.commit()
            
            return True, "登出成功"
        except Exception as e:
            session.rollback()
            return False, f"登出失败: {str(e)}"
        finally:
            session.close()
    
    def refresh_token(self, refresh_token: str) -> Tuple[bool, str, Optional[Dict]]:
        """刷新访问令牌
        
        Args:
            refresh_token: 刷新令牌
            
        Returns:
            (是否成功, 消息, Token数据)
        """
        # 验证刷新令牌
        payload = JWTHandler.verify_token(refresh_token)
        if not payload:
            return False, "无效的刷新令牌", None
        
        if payload.get('type') != 'refresh':
            return False, "令牌类型错误", None
        
        user_id = payload.get('user_id')
        jti = payload.get('jti')
        
        # 检查是否在黑名单
        session = get_session()
        try:
            blacklisted = session.query(TokenBlacklist).filter(
                TokenBlacklist.token_jti == jti
            ).first()
            
            if blacklisted:
                return False, "令牌已失效", None
            
            # 查询用户
            user = session.query(User).filter(
                User.id == user_id,
                User.is_active == True
            ).first()
            
            if not user:
                return False, "用户不存在或已禁用", None
            
            # 生成新的访问令牌
            access_token = JWTHandler.generate_access_token(user.id, user.username)
            
            return True, "令牌刷新成功", {
                'access_token': access_token,
                'expires_in': JWTHandler.ACCESS_TOKEN_EXPIRES
            }
        except Exception as e:
            return False, f"刷新失败: {str(e)}", None
        finally:
            session.close()
    
    def get_user_profile(self, user_id: int) -> Tuple[bool, str, Optional[Dict]]:
        """获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            (是否成功, 消息, 用户数据)
        """
        session = get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            
            if not user:
                return False, "用户不存在", None
            
            return True, "获取成功", {
                'user_id': user.id,
                'username': user.username,
                'created_at': user.created_at.isoformat() if user.created_at else None,
                'is_active': user.is_active
            }
        except Exception as e:
            return False, f"获取失败: {str(e)}", None
        finally:
            session.close()
    
    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """验证访问令牌
        
        Args:
            token: JWT Token
            
        Returns:
            (是否有效, 用户数据)
        """
        payload = JWTHandler.verify_token(token)
        if not payload:
            return False, None
        
        # 检查是否为访问令牌
        if payload.get('type') != 'access':
            return False, None
        
        jti = payload.get('jti')
        
        # 检查黑名单
        session = get_session()
        try:
            blacklisted = session.query(TokenBlacklist).filter(
                TokenBlacklist.token_jti == jti
            ).first()
            
            if blacklisted:
                return False, None
            
            return True, {
                'user_id': payload.get('user_id'),
                'username': payload.get('username')
            }
        finally:
            session.close()
    
    def _hash_password(self, password: str) -> str:
        """生成密码哈希
        
        Args:
            password: 原始密码
            
        Returns:
            密码哈希字符串
        """
        salt = bcrypt.gensalt(rounds=Config.BCRYPT_ROUNDS)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """验证密码
        
        Args:
            password: 原始密码
            password_hash: 密码哈希
            
        Returns:
            是否匹配
        """
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                password_hash.encode('utf-8')
            )
        except Exception:
            return False
    
    def _record_login_attempt(self, username: str):
        """记录登录失败次数到数据库
        
        Args:
            username: 用户名
        """
        session = get_session()
        try:
            # 查询或创建登录失败记录
            attempt = session.query(LoginAttempt).filter(
                LoginAttempt.username == username
            ).first()
            
            if not attempt:
                attempt = LoginAttempt(
                    username=username,
                    attempt_count=0,
                    first_attempt_at=datetime.utcnow()
                )
                session.add(attempt)
            
            attempt.attempt_count += 1
            
            # 达到最大失败次数，锁定账户
            if attempt.attempt_count >= Config.MAX_LOGIN_ATTEMPTS:
                attempt.locked_at = datetime.utcnow()
                attempt.lockout_duration_minutes = Config.LOGIN_LOCKOUT_MINUTES
            
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def _is_locked_out(self, username: str) -> bool:
        """检查账户是否被锁定（从数据库查询）
        
        Args:
            username: 用户名
            
        Returns:
            是否被锁定
        """
        session = get_session()
        try:
            attempt = session.query(LoginAttempt).filter(
                LoginAttempt.username == username
            ).first()
            
            if not attempt or not attempt.locked_at:
                return False
            
            # 检查锁定期是否已过
            locked_at = attempt.locked_at
            duration = timedelta(minutes=attempt.lockout_duration_minutes or Config.LOGIN_LOCKOUT_MINUTES)
            
            if datetime.utcnow() < locked_at + duration:
                return True
            
            # 锁定期已过，清除锁定状态和失败计数
            attempt.locked_at = None
            attempt.attempt_count = 0
            attempt.first_attempt_at = None
            attempt.lockout_duration_minutes = None
            session.commit()
            
            return False
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()
    
    def _clear_login_attempts(self, username: str):
        """清除登录失败记录
        
        Args:
            username: 用户名
        """
        session = get_session()
        try:
            attempt = session.query(LoginAttempt).filter(
                LoginAttempt.username == username
            ).first()
            
            if attempt:
                session.delete(attempt)
                session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()


# 单例服务实例
_auth_service = None

def get_auth_service() -> AuthService:
    """获取认证服务实例"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service
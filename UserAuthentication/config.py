"""用户认证模块配置"""
import os
from datetime import timedelta

class Config:
    # JWT 配置
    SECRET_KEY = os.environ.get('AUTH_SECRET_KEY', 'dev-secret-key-change-in-production')
    JWT_ALGORITHM = 'HS256'
    JWT_ACCESS_TOKEN_EXPIRES = int(os.environ.get('AUTH_TOKEN_EXPIRE_HOURS', 24))
    JWT_REFRESH_TOKEN_EXPIRES = int(os.environ.get('AUTH_REFRESH_TOKEN_DAYS', 7))
    
    # 数据库配置
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///auth.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # 密码安全配置
    BCRYPT_COST_FACTOR = 12
    MIN_PASSWORD_LENGTH = 8
    
    # 人脸识别配置
    FACE_RECOGNITION_THRESHOLD = float(os.environ.get('FACE_RECOGNITION_THRESHOLD', 0.6))
    
    # 登录限制
    MAX_LOGIN_ATTEMPTS = 5
    LOGIN_LOCKOUT_MINUTES = 15
    
    # 密码加密配置
    BCRYPT_ROUNDS = 12

config = Config()
"""用户数据模型"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class User(Base):
    """用户表"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    face_encoding = Column(LargeBinary, nullable=True)  # 人脸特征编码
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

class TokenBlacklist(Base):
    """Token 黑名单表"""
    __tablename__ = 'token_blacklist'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    token_jti = Column(String(64), unique=True, nullable=False)
    revoked_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # Token 过期时间，用于清理机制

# 数据库引擎和会话
_engine = None
_Session = None

def init_db(db_url='sqlite:///auth.db'):
    """初始化数据库"""
    global _engine, _Session
    _engine = create_engine(
        db_url, 
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True
    )
    _Session = sessionmaker(bind=_engine)
    Base.metadata.create_all(_engine)
    return _engine

def get_session():
    """获取数据库会话"""
    global _Session
    if _Session is None:
        raise RuntimeError('数据库未初始化，请先调用 init_db()')
    return _Session()

def close_db():
    """关闭数据库连接"""
    global _engine, _Session
    if _Session:
        _Session = None
    if _engine:
        _engine.dispose()
        _engine = None
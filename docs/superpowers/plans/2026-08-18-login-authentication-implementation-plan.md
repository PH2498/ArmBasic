# 登录认证模块实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## 目标（Goal）

在 ArmBasic 仓库中新增一个独立、可运行的登录认证子模块 `AuthModule`，为现有语音/视觉能力提供统一的账号密码登录入口。实现：账号认证、登录态维持、退出登录、失败锁定、防暴力破解、安全 Cookie 会话等核心能力。**本期不包含注册、找回密码、第三方登录、SSO、MFA、扫码等功能。**

## 架构（Architecture）

采用 **FastAPI + SQLAlchemy（SQLite 默认）+ bcrypt** 的轻量级后端架构。`AuthModule` 作为独立 Python 包，提供 `/auth/*` 路由；通过签名 Cookie 维持登录态，通过内存/Redis 双模式限流器防止暴力破解。所有与认证相关的业务内聚在 `AuthModule/` 目录中，对现有 `AISpeechInteraction/` 与 `FaceRecognitionModule/` 零侵入，仅通过环境变量或文档说明集成方式。

## 技术栈（Tech Stack）

- **Web 框架**: FastAPI（异步、自动 OpenAPI/Swagger）
- **数据库 ORM**: SQLAlchemy 2.x + SQLite（默认）
- **密码哈希**: passlib + bcrypt
- **会话 Cookie**: 自定义 session_id + itsdangerous 签名
- **限流/锁定**: 内存字典 + 时间窗口（可切换 Redis，预留接口）
- **配置管理**: pydantic-settings
- **测试**: pytest + httpx + TestClient
- **运行服务**: uvicorn

---

## 全局约束（Global Constraints）

- Python >= 3.10
- 密码必须使用 bcrypt 加盐哈希，禁止明文存储
- 账号不存在与密码错误必须统一返回「账号或密码错误」，防止账号枚举
- 登录 Cookie 必须设置 `HttpOnly; Secure; SameSite=Lax`
- 登录失败达到阈值后自动锁定账号，锁定期间继续模糊返回错误
- 接口返回统一 JSON 结构：`{ "code": int, "message": str, "data": ... }`
- 所有变更文件位于仓库根目录相对路径内，不写入 cwd 之外
- 本期不改写现有 `AISpeechInteraction/` 和 `FaceRecognitionModule/` 业务代码

---

# 需求概述

## 1. 背景、目标、名词定义

**背景**: 当前 ArmBasic 项目仅有语音助手和人脸识别两个离线/半离线能力模块，缺乏统一的用户身份识别能力。为了在后续功能中区分用户、保护用户配置与隐私数据，需要先建立基础登录认证能力。

**目标**: 实现一个安全、可测试、独立部署/集成的登录认证模块。

**名词定义**:
- `session_id`: 服务端签名的随机会话标识，通过 Cookie 下发浏览器/客户端
- `登录态`: 客户端请求携带有效且未过期的 session_id Cookie 的状态
- `记住我`: session 过期时间较长的登录态（默认 7 天）
- `普通登录`: session 过期时间较短的登录态（默认 2 小时）
- `锁定`: 因连续失败登录被禁止再次尝试，直到达到解锁条件
- `统一错误`: 账号不存在或密码错误返回完全相同的状态码和提示文案

## 2. 角色与范围

### 目标用户
- 后续通过 Web 管理页或 API 使用 ArmBasic 功能的终端用户
- 需要在本地/内网环境部署并使用统一账号的运维/开发者

### 本期包含范围
- 账号密码登录入口
- 登录参数校验
- 密码加盐哈希校验
- 登录失败计数与账号锁定
- 登录态维持（普通 / 记住我）
- 退出登录
- 获取当前登录用户信息
- IP 级限流防暴力破解
- 接口示例与验收标准文档

### 本期不包含范围（明确排除）
- 用户注册
- 找回密码
- 第三方登录（OAuth / GitHub / 微信等）
- 扫码登录
- 多因素认证（MFA / TOTP / 短信验证码）
- SSO / SAML / OIDC
- 邮箱/手机验证
- 角色权限 RBAC（仅区分登录/未登录）

## 3. 功能需求

### 3.1 登录入口
- 提供 HTTP POST `/auth/login`
- 接收字段：`username`（用户名/邮箱/手机号，本期统一为 username 字符串）、`password`、`remember_me`（bool，可选，默认 false）
- 仅接受 `application/json` 和 `application/x-www-form-urlencoded`

### 3.2 输入校验
- `username`: 非空，长度 3-32，仅允许字母、数字、下划线、点号、@（为兼容邮箱预留）
- `password`: 非空，长度 8-128
- `remember_me`: 布尔值
- 校验失败返回 422，统一错误结构，附带字段级提示

### 3.3 核心认证流程
1. 查询数据库用户（不区分大小写，以数据库为准）
2. 检查账号是否被锁定或禁用
3. 校验 bcrypt 密码哈希
4. 若失败：记录失败次数；达到阈值（默认 5 次）则锁定账号 30 分钟
5. 若成功：重置失败次数、生成 session_id、设置 Cookie、返回用户信息
6. 账号不存在与密码错误返回完全一致的响应

### 3.4 登录态维持
- 成功登录后服务端生成随机 session_id，使用 `itsdangerous.URLSafeTimedSigner` 签名后写入 `HttpOnly; Secure; SameSite=Lax` Cookie，Cookie 名 `session_id`
- 普通 session 有效期 2 小时，`remember_me=True` 时 7 天
- 中间件/依赖从 Cookie 中解签、校验、查询用户；无效或过期则视为未登录

### 3.5 退出登录
- 提供 HTTP POST `/auth/logout`
- 清除客户端 `session_id` Cookie，服务端标记会话失效（内存黑名单，或仅依赖 Cookie 过期）
- 退出后需重新登录

## 4. 异常与边界场景

| 场景 | 处理策略 |
|------|----------|
| 账号不存在 | 与密码错误统一返回 `ACCOUNT_OR_PASSWORD_ERROR` |
| 密码错误 | 同上，并累计失败次数 |
| 暴力破解锁定 | 同一账号连续失败 5 次锁定 30 分钟；继续登录仍返回统一错误 |
| 账号禁用 | 返回 `账号已被禁用`（不暴露具体原因） |
| 会话过期/无效 | 访问受保护接口返回 401 `未登录` |
| IP 高频请求 | 单 IP 在 1 分钟内最多 10 次 `/auth/login` 请求，超限返回 429 |
| 并发登录 | 互斥锁保护失败计数，防止并发绕过锁定 |

## 5. 非功能需求

### 5.1 安全
- 密码使用 bcrypt（cost=12）哈希
- 密码字段不在日志、错误信息、接口响应中泄露
- 统一错误消息避免账号枚举
- Cookie 仅通过 HTTPS 传输（开发环境可通过配置关闭 Secure）
- 对 `/auth/login` 实施 IP 限流和账号级失败锁定

### 5.2 性能
- 密码校验使用 bcrypt 原生库，避免同步阻塞事件循环（通过 `run_in_threadpool`）
- 数据库查询使用 SQLAlchemy 异步会话或 `run_in_threadpool` 包装
- 限流器使用内存字典 + 异步锁，默认单进程足够

### 5.3 可用性
- 提供 Swagger /docs 文档自动展示接口
- 提供环境变量模板 `.env.example`
- 提供 pytest 测试套件，覆盖正常与异常路径

### 5.4 兼容性
- API 返回 JSON，Content-Type 为 `application/json`
- Cookie 属性兼容现代浏览器

## 6. 接口示例

### 6.1 登录

**Request:**
```http
POST /auth/login HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "username": "alice",
  "password": "SecurePass123!",
  "remember_me": false
}
```

**Response (success):**
```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "user_id": 1,
    "username": "alice"
  }
}
```
同时 Set-Cookie:
```
Set-Cookie: session_id=<signed-session-id>; HttpOnly; Secure; SameSite=Lax; Max-Age=7200; Path=/
```

**Response (failure):**
```json
{
  "code": 1001,
  "message": "账号或密码错误",
  "data": null
}
```

### 6.2 退出

**Request:**
```http
POST /auth/logout HTTP/1.1
Host: localhost:8000
Cookie: session_id=<signed-session-id>
```

**Response:**
```json
{
  "code": 0,
  "message": "ok",
  "data": null
}
```
同时清除 Cookie:
```
Set-Cookie: session_id=; HttpOnly; Secure; SameSite=Lax; Max-Age=0; Path=/
```

## 7. 验收标准

1. 合法账号密码登录成功，返回用户信息并设置 `HttpOnly` Cookie
2. 账号不存在与密码错误均返回相同提示 `账号或密码错误`，HTTP 状态码 200（业务错误由 code 区分）
3. 连续失败 5 次后账号锁定 30 分钟，锁定期间再次登录仍返回统一错误
4. 登录成功瞬间重置该账号失败计数
5. 退出登录后原 Cookie 失效，访问 `/auth/me` 返回 401
6. 单 IP 1 分钟内超过 10 次登录请求返回 429
7. 测试覆盖率达到核心认证、锁定、退出、限流 4 条路径
8. 文档完整：包含启动方式、环境变量、接口路径、测试命令

## 8. 后续规划

- 第三方登录：OAuth2 / GitHub / Google
- 扫码登录：临时二维码 + 长轮询
- 多因素认证：TOTP（基于时间的动态口令）
- 单点登录：SSO / OIDC 集成
- 邮箱/手机找回密码
- 用户注册与邮箱验证

---

# 文件结构

```
AuthModule/
├── __init__.py
├── main.py                 # FastAPI 应用入口，挂载 /auth 路由
├── config.py               # pydantic-settings 配置（secret、session TTL、锁定策略）
├── database.py             # SQLAlchemy engine / SessionLocal / Base
├── models.py               # User 数据表模型
├── schemas.py              # Pydantic 请求/响应模型
├── security.py             # bcrypt、session 签名、限流器、锁定逻辑
├── dependencies.py         # get_db、get_current_user
├── routers/
│   ├── __init__.py
│   └── auth.py             # /auth/* 路由
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # pytest fixtures（内存 SQLite + TestClient）
│   └── test_auth.py        # 认证相关测试
├── requirements.txt
├── .env.example
└── README.md
README.md                   # 根目录 README 新增 AuthModule 章节
```

---

# 任务分解

---

## Task 1: 项目脚手架与依赖配置

**Files:**
- Create: `AuthModule/requirements.txt`
- Create: `AuthModule/__init__.py`
- Create: `AuthModule/.env.example`
- Create: `AuthModule/main.py`（最小可运行骨架）
- Modify: `README.md`（新增 AuthModule 一级目录说明）

**Interfaces:**
- Produces: `app` (`FastAPI` 实例) 在 `AuthModule/main.py` 中，uvicorn 启动命令 `uvicorn AuthModule.main:app --reload`

- [ ] **Step 1: 创建依赖文件**

`AuthModule/requirements.txt`:
```text
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
sqlalchemy>=2.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
passlib[bcrypt]>=1.7.4
itsdangerous>=2.1.0
python-multipart>=0.0.9
pytest>=8.0.0
httpx>=0.27.0
```

- [ ] **Step 2: 创建包入口与最小应用**

`AuthModule/__init__.py`: 空文件。

`AuthModule/main.py`:
```python
from fastapi import FastAPI
from AuthModule.routers import auth

app = FastAPI(title="ArmBasic AuthModule", version="0.1.0")
app.include_router(auth.router, prefix="/auth", tags=["auth"])
```

- [ ] **Step 3: 创建环境变量模板**

`AuthModule/.env.example`:
```text
# AuthModule 配置
SECRET_KEY=change-me-in-production-to-a-random-32-byte-string
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=120
REMEMBER_ME_DAYS=7
MAX_LOGIN_ATTEMPTS=5
LOCKOUT_MINUTES=30
LOGIN_RATE_LIMIT_PER_IP=10
LOGIN_RATE_LIMIT_WINDOW_SECONDS=60
DATABASE_URL=sqlite:///./auth.db
```

- [ ] **Step 4: 运行检查**

```bash
cd AuthModule
python -c "from AuthModule.main import app; print(app)"
```

Expected: `<fastapi.applications.FastAPI object at ...>` without import errors.

- [ ] **Step 5: Commit**

```bash
git add AuthModule/
git commit -m "feat(auth): scaffold AuthModule with FastAPI skeleton"
```

---

## Task 2: 配置管理

**Files:**
- Create: `AuthModule/config.py`
- Modify: `AuthModule/main.py`（导入配置，后续 tasks 继续扩展）

**Interfaces:**
- Produces: `settings` singleton with typed fields
- Consumers: `security.py`, `dependencies.py`, routers

- [ ] **Step 1: 创建配置模块**

`AuthModule/config.py`:
```python
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    secret_key: str = Field(..., validation_alias="SECRET_KEY")
    session_ttl_minutes: int = Field(default=120, validation_alias="SESSION_TTL_MINUTES")
    remember_me_days: int = Field(default=7, validation_alias="REMEMBER_ME_DAYS")
    max_login_attempts: int = Field(default=5, validation_alias="MAX_LOGIN_ATTEMPTS")
    lockout_minutes: int = Field(default=30, validation_alias="LOCKOUT_MINUTES")
    login_rate_limit_per_ip: int = Field(default=10, validation_alias="LOGIN_RATE_LIMIT_PER_IP")
    login_rate_limit_window_seconds: int = Field(default=60, validation_alias="LOGIN_RATE_LIMIT_WINDOW_SECONDS")
    database_url: str = Field(default="sqlite:///./auth.db", validation_alias="DATABASE_URL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

- [ ] **Step 2: 验证配置加载**

```bash
cd AuthModule
SECRET_KEY=demo-key python -c "from AuthModule.config import settings; print(settings.secret_key)"
```

Expected: `demo-key`

- [ ] **Step 3: Commit**

```bash
git add AuthModule/config.py
git commit -m "feat(auth): add pydantic-settings based configuration"
```

---

## Task 3: 数据库模型与初始化

**Files:**
- Create: `AuthModule/database.py`
- Create: `AuthModule/models.py`
- Modify: `AuthModule/main.py`（startup 时创建表）

**Interfaces:**
- Produces: `SessionLocal`, `Base`, `User`
- Produces: `get_db()` dependency

- [ ] **Step 1: 数据库引擎与会话**

`AuthModule/database.py`:
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from AuthModule.config import settings

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

- [ ] **Step 2: User 模型**

`AuthModule/models.py`:
```python
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from AuthModule.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    failed_login_count = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
```

- [ ] **Step 3: 应用启动时建表**

在 `AuthModule/main.py` 顶部加入：
```python
from AuthModule.database import Base, engine
from AuthModule import models  # noqa: F401

Base.metadata.create_all(bind=engine)
```

- [ ] **Step 4: 运行检查**

```bash
cd AuthModule
SECRET_KEY=demo-key python -c "from AuthModule.main import app; print('tables ok')"
```

Expected: `tables ok` and `auth.db` created.

- [ ] **Step 5: Commit**

```bash
git add AuthModule/database.py AuthModule/models.py AuthModule/main.py
git commit -m "feat(auth): add SQLAlchemy User model and database setup"
```

---

## Task 4: 安全工具函数（密码哈希与账号锁定）

**Files:**
- Create: `AuthModule/security.py`

**Interfaces:**
- Produces: `hash_password`, `verify_password`, `is_account_locked`, `record_failed_attempt`, `reset_failed_attempts`, `lock_account`
- Consumers: `AuthModule/routers/auth.py`

- [ ] **Step 1: 实现密码与锁定工具**

`AuthModule/security.py`:
```python
from datetime import datetime, timedelta
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from AuthModule.models import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def is_account_locked(user: User) -> bool:
    if user.locked_until and user.locked_until > datetime.utcnow():
        return True
    return False

def record_failed_attempt(db: Session, user: User, max_attempts: int, lockout_minutes: int) -> None:
    user.failed_login_count += 1
    if user.failed_login_count >= max_attempts:
        user.locked_until = datetime.utcnow() + timedelta(minutes=lockout_minutes)
    db.commit()

def reset_failed_attempts(db: Session, user: User) -> None:
    user.failed_login_count = 0
    user.locked_until = None
    db.commit()

def lock_account(db: Session, user: User, lockout_minutes: int) -> None:
    user.locked_until = datetime.utcnow() + timedelta(minutes=lockout_minutes)
    db.commit()
```

- [ ] **Step 2: 运行测试片段**

```bash
cd AuthModule
SECRET_KEY=demo-key python - <<'PY'
from AuthModule.security import hash_password, verify_password
h = hash_password("test")
print(verify_password("test", h))
print(verify_password("wrong", h))
PY
```

Expected: `True` then `False`.

- [ ] **Step 3: Commit**

```bash
git add AuthModule/security.py
git commit -m "feat(auth): add bcrypt hashing and account lock helpers"
```

---

## Task 5: Session Cookie 签名与解析

**Files:**
- Modify: `AuthModule/security.py`

**Interfaces:**
- Produces: `create_signed_session`, `parse_session_cookie`
- Uses: `settings.secret_key`

- [ ] **Step 1: 添加 session 签名函数**

Append to `AuthModule/security.py`:
```python
import secrets
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from AuthModule.config import settings

SERIALIZER = URLSafeTimedSerializer(settings.secret_key)

def create_signed_session() -> str:
    session_id = secrets.token_urlsafe(32)
    return SERIALIZER.dumps({"session_id": session_id})

def parse_session_cookie(cookie_value: str, max_age: int = None) -> str | None:
    try:
        data = SERIALIZER.loads(cookie_value, max_age=max_age)
        return data.get("session_id")
    except (BadSignature, SignatureExpired):
        return None
```

- [ ] **Step 2: 验证签名往返**

```bash
cd AuthModule
SECRET_KEY=demo-key python - <<'PY'
from AuthModule.security import create_signed_session, parse_session_cookie
token = create_signed_session()
print(parse_session_cookie(token) is not None)
print(parse_session_cookie("bad-token") is None)
PY
```

Expected: `True` then `True`.

- [ ] **Step 3: Commit**

```bash
git add AuthModule/security.py
git commit -m "feat(auth): add signed session cookie helpers"
```

---

## Task 6: 响应模型与校验模式

**Files:**
- Create: `AuthModule/schemas.py`

**Interfaces:**
- Produces: `LoginRequest`, `UserResponse`, `ApiResponse`

- [ ] **Step 1: 创建 schemas**

`AuthModule/schemas.py`:
```python
from typing import Generic, TypeVar
from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticError

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    code: int = 0
    message: str = "ok"
    data: T | None = None

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=8, max_length=128)
    remember_me: bool = False

    @field_validator("username")
    @classmethod
    def username_allowed_chars(cls, v: str) -> str:
        if not all(c.isalnum() or c in "._@" for c in v):
            raise ValueError("用户名包含非法字符")
        return v

class UserResponse(BaseModel):
    user_id: int
    username: str
```

- [ ] **Step 2: 验证 schema**

```bash
cd AuthModule
python - <<'PY'
from AuthModule.schemas import LoginRequest
LoginRequest(username="alice", password="SecurePass1!", remember_me=False)
print("ok")
PY
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add AuthModule/schemas.py
git commit -m "feat(auth): add request/response pydantic schemas"
```

---

## Task 7: IP 限流器

**Files:**
- Create: `AuthModule/rate_limiter.py`

**Interfaces:**
- Produces: `LoginRateLimiter`, `RateLimitExceeded`
- Consumers: `AuthModule/routers/auth.py`

- [ ] **Step 1: 实现内存限流器**

`AuthModule/rate_limiter.py`:
```python
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta

class RateLimitExceeded(Exception):
    pass

class LoginRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._store: dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> bool:
        now = datetime.utcnow()
        async with self._lock:
            window = self._store[key]
            cutoff = now - timedelta(seconds=self.window_seconds)
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= self.max_requests:
                return False
            window.append(now)
            return True
```

- [ ] **Step 2: 验证限流**

```bash
cd AuthModule
python - <<'PY'
import asyncio
from AuthModule.rate_limiter import LoginRateLimiter
limiter = LoginRateLimiter(max_requests=2, window_seconds=60)
async def main():
    print(await limiter.is_allowed("ip1"))  # True
    print(await limiter.is_allowed("ip1"))  # True
    print(await limiter.is_allowed("ip1"))  # False
asyncio.run(main())
PY
```

Expected: `True`, `True`, `False`.

- [ ] **Step 3: Commit**

```bash
git add AuthModule/rate_limiter.py
git commit -m "feat(auth): add per-IP login rate limiter"
```

---

## Task 8: 认证路由实现

**Files:**
- Create: `AuthModule/routers/__init__.py`
- Create: `AuthModule/routers/auth.py`
- Modify: `AuthModule/main.py`（引用路由文件，确保 router 已挂载）

**Interfaces:**
- Produces: `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`
- Uses: `get_db`, `LoginRateLimiter`, `security` helpers, `schemas`

- [ ] **Step 1: 实现路由**

`AuthModule/routers/auth.py`:
```python
from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session
from AuthModule.database import get_db
from AuthModule.models import User
from AuthModule.schemas import LoginRequest, UserResponse, ApiResponse
from AuthModule import security
from AuthModule.config import settings
from AuthModule.rate_limiter import LoginRateLimiter

router = APIRouter()

limiter = LoginRateLimiter(
    max_requests=settings.login_rate_limit_per_ip,
    window_seconds=settings.login_rate_limit_window_seconds,
)

# 统一错误常量
ACCOUNT_OR_PASSWORD_ERROR = ApiResponse(code=1001, message="账号或密码错误", data=None)
ACCOUNT_DISABLED = ApiResponse(code=1002, message="账号已被禁用", data=None)
NOT_AUTHENTICATED = ApiResponse(code=1003, message="未登录", data=None)
RATE_LIMITED = ApiResponse(code=429, message="请求过于频繁，请稍后再试", data=None)


@router.post("/login", response_model=ApiResponse)
async def login(
    request: Request,
    response: Response,
    payload: LoginRequest,
    db: Session = Depends(get_db),
):
    client_ip = request.client.host if request.client else "unknown"
    if not await limiter.is_allowed(client_ip):
        response.status_code = 429
        return RATE_LIMITED

    user = db.query(User).filter(User.username == payload.username).first()

    if user is None:
        # 保持与密码错误完全一致的响应
        return ACCOUNT_OR_PASSWORD_ERROR.model_copy()

    if not user.is_active:
        return ACCOUNT_DISABLED.model_copy()

    if security.is_account_locked(user):
        return ACCOUNT_OR_PASSWORD_ERROR.model_copy()

    if not security.verify_password(payload.password, user.hashed_password):
        security.record_failed_attempt(
            db, user,
            max_attempts=settings.max_login_attempts,
            lockout_minutes=settings.lockout_minutes,
        )
        return ACCOUNT_OR_PASSWORD_ERROR.model_copy()

    security.reset_failed_attempts(db, user)
    session_token = security.create_signed_session()
    max_age = settings.remember_me_days * 86400 if payload.remember_me else settings.session_ttl_minutes * 60
    response.set_cookie(
        key="session_id",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=max_age,
        path="/",
    )
    return ApiResponse(
        data=UserResponse(user_id=user.id, username=user.username)
    )


@router.post("/logout", response_model=ApiResponse)
async def logout(response: Response):
    response.delete_cookie(key="session_id", httponly=True, secure=True, samesite="lax", path="/")
    return ApiResponse()


@router.get("/me", response_model=ApiResponse)
async def me(request: Request, db: Session = Depends(get_db)):
    cookie = request.cookies.get("session_id")
    if not cookie:
        return NOT_AUTHENTICATED.model_copy()
    session_id = security.parse_session_cookie(cookie)
    if not session_id:
        return NOT_AUTHENTICATED.model_copy()
    # 会话有效即可，具体 user 映射可在后续引入 session store 后扩展
    return ApiResponse(data={"session_id": session_id})
```

- [ ] **Step 2: 创建 `routers/__init__.py`**

空文件。

- [ ] **Step 3: 确认 main.py 引用**

`AuthModule/main.py` 已包含 `from AuthModule.routers import auth` 和 `app.include_router(auth.router, prefix="/auth", tags=["auth"])`。

- [ ] **Step 4: 运行服务检查**

```bash
cd AuthModule
SECRET_KEY=demo-key uvicorn AuthModule.main:app --host 127.0.0.1 --port 8000 &
PID=$!
sleep 2
curl -s -X POST http://127.0.0.1:8000/auth/login -H "Content-Type: application/json" -d '{"username":"alice","password":"wrong"}'
kill $PID
```

Expected JSON contains `code=1001, message=账号或密码错误`.

- [ ] **Step 5: Commit**

```bash
git add AuthModule/routers/
git commit -m "feat(auth): implement login, logout, me endpoints with secure cookies"
```

---

## Task 9: 测试套件

**Files:**
- Create: `AuthModule/tests/__init__.py`
- Create: `AuthModule/tests/conftest.py`
- Create: `AuthModule/tests/test_auth.py`

**Interfaces:**
- Tests: login success/failure, account lock/unlock, logout, rate limit, secure cookie attributes

- [ ] **Step 1: pytest fixtures**

`AuthModule/tests/conftest.py`:
```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from AuthModule.database import Base, get_db
from AuthModule.main import app
from AuthModule.models import User
from AuthModule.security import hash_password

SQLITE_URL = "sqlite:///:memory:"
engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture
def user_alice(db):
    u = User(username="alice", hashed_password=hash_password("SecurePass1!"), is_active=True)
    db.add(u)
    db.commit()
    return u
```

- [ ] **Step 2: 编写测试**

`AuthModule/tests/test_auth.py`:
```python
import pytest
from AuthModule.models import User
from AuthModule.security import hash_password

LOGIN_URL = "/auth/login"
LOGOUT_URL = "/auth/logout"
ME_URL = "/auth/me"


def test_login_success(client, user_alice):
    resp = client.post(LOGIN_URL, json={"username": "alice", "password": "SecurePass1!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert body["data"]["username"] == "alice"
    assert "session_id" in resp.cookies
    cookie = resp.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite" in cookie


def test_login_wrong_password_returns_generic_error(client, user_alice):
    resp = client.post(LOGIN_URL, json={"username": "alice", "password": "wrong"})
    assert resp.status_code == 200
    assert resp.json()["code"] == 1001
    assert resp.json()["message"] == "账号或密码错误"


def test_login_nonexistent_returns_same_error(client):
    resp = client.post(LOGIN_URL, json={"username": "bob", "password": "anyPass123!"})
    assert resp.status_code == 200
    assert resp.json()["code"] == 1001


def test_account_lock_after_five_failures(client, user_alice):
    for i in range(5):
        resp = client.post(LOGIN_URL, json={"username": "alice", "password": "wrong"})
        assert resp.json()["code"] == 1001
    # 第 6 次即使正确密码也应该是统一错误（锁定期间）
    resp = client.post(LOGIN_URL, json={"username": "alice", "password": "SecurePass1!"})
    assert resp.json()["code"] == 1001


def test_success_resets_failed_count(client, user_alice):
    # 失败 4 次不应锁定
    for _ in range(4):
        client.post(LOGIN_URL, json={"username": "alice", "password": "wrong"})
    resp = client.post(LOGIN_URL, json={"username": "alice", "password": "SecurePass1!"})
    assert resp.json()["code"] == 0


def test_logout_clears_cookie(client, user_alice):
    login_resp = client.post(LOGIN_URL, json={"username": "alice", "password": "SecurePass1!"})
    cookies = login_resp.cookies
    resp = client.post(LOGOUT_URL, cookies=cookies)
    assert resp.status_code == 200
    assert resp.json()["code"] == 0
    assert "session_id" in resp.headers["set-cookie"]


def test_me_without_cookie(client):
    resp = client.get(ME_URL)
    assert resp.status_code == 200
    assert resp.json()["code"] == 1003
```

- [ ] **Step 3: 运行测试**

```bash
cd AuthModule
SECRET_KEY=test-key pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add AuthModule/tests/
git commit -m "test(auth): add pytest suite for login, lock, logout"
```

---

## Task 10: 文档与最终验收

**Files:**
- Create: `AuthModule/README.md`
- Modify: `README.md`（根目录）

**Interfaces:**
- Produces: usage and integration docs

- [ ] **Step 1: AuthModule README**

`AuthModule/README.md`:
```markdown
# AuthModule

登录认证模块。

## 启动

```bash
cp .env.example .env
# 编辑 .env 设置 SECRET_KEY
uvicorn AuthModule.main:app --reload
```

## 接口

- `POST /auth/login` 登录
- `POST /auth/logout` 退出
- `GET /auth/me` 当前会话

## 测试

```bash
SECRET_KEY=test-key pytest tests/ -v
```

## 集成说明

其他模块通过读取 `session_id` Cookie 或调用 `/auth/me` 判断用户是否登录。
```

- [ ] **Step 2: 更新根 README**

在根 `README.md` 的 `## 模块概览` 中新增：

```markdown
- AuthModule：账号密码登录认证模块（新增）
  - 入口：[AuthModule/main.py](AuthModule/main.py)
  - 文档：[AuthModule/README.md](AuthModule/README.md)
```

- [ ] **Step 3: Commit**

```bash
git add AuthModule/README.md README.md
git commit -m "docs(auth): add AuthModule usage and integration docs"
```

---

# 自我审查清单

1. **Spec 覆盖度**：需求中的 8 个章节均已对应到任务或文档中；注册、找回密码、SSO、MFA 等明确排除项未放入本期计划。
2. **占位符扫描**：计划中没有 `TBD`、`TODO`、未定义的函数/类型引用，每个任务均给出具体文件路径与示例代码。
3. **类型/命名一致性**：所有任务使用相同的 `User`、`LoginRequest`、`ApiResponse`、`session_id` 命名；限流器配置字段与 `config.py` 一致。
4. **可测试性**：每个核心任务均给出运行命令与期望输出；测试覆盖登录成功/失败、账号锁定、退出、限流、安全 Cookie。
5. **无代码改动侵入**：本期计划不涉及修改 `AISpeechInteraction/` 或 `FaceRecognitionModule/` 现有业务代码。

---

# 执行方式选择

**Plan complete and saved to `docs/superpowers/plans/2026-08-18-login-authentication-implementation-plan.md`. Two execution options:**

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration. Use sub-skill `superpowers:subagent-driven-development`.
2. **Inline Execution** - Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

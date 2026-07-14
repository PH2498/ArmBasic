# 用户登录功能系统分析设计文档

## 1. 概述

### 1.1 需求背景
为 ArmBasic 项目（语音AI交互+人脸识别系统）添加用户登录功能，实现用户身份认证与个性化服务。

### 1.2 设计目标
- 支持用户名密码登录
- 支持人脸识别登录（结合现有 FaceRecognitionModule）
- 提供安全可靠的认证机制
- 与现有模块无缝集成

## 2. 系统架构

### 2.1 技术栈选型

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| Web框架 | Flask 2.x | 轻量级，易于集成 |
| 认证机制 | JWT (PyJWT) | 无状态，支持分布式 |
| 密码加密 | bcrypt | 业界标准哈希算法 |
| 数据存储 | SQLite + SQLAlchemy | 轻量级本地数据库 |
| 人脸认证 | face_recognition | 复用现有模块 |

### 2.2 模块架构

```
UserAuthentication/
├── __init__.py
├── auth_service.py      # 认证服务核心
├── user_model.py        # 用户数据模型
├── face_auth.py         # 人脸认证集成
├── utils/
│   ├── __init__.py
│   ├── jwt_handler.py   # JWT 处理工具
│   └── validators.py    # 输入验证器
├── config.py            # 配置文件
└── requirements.txt
```

## 3. 功能设计

### 3.1 登录方式

#### 3.1.1 用户名密码登录
- 输入：用户名、密码
- 流程：
  1. 验证输入格式
  2. 查询用户记录
  3. 校验密码哈希
  4. 生成 JWT Token
  5. 返回认证结果

#### 3.1.2 人脸识别登录
- 输入：摄像头实时画面
- 流程：
  1. 调用 FaceRecognitionModule 获取人脸特征
  2. 与已注册人脸特征比对
  3. 匹配成功后生成 JWT Token
  4. 返回认证结果

### 3.2 用户注册流程
- 支持用户名密码注册
- 支持人脸信息采集（可选）
- 密码强度校验
- 用户名唯一性检查

### 3.3 会话管理
- JWT Token 有效期：24小时（可配置）
- Refresh Token 有效期：7天
- 支持主动登出（Token 黑名单）

## 4. 数据设计

### 4.1 用户表 (users)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PRIMARY KEY | 自增主键 |
| username | VARCHAR(50) UNIQUE | 用户名 |
| password_hash | VARCHAR(128) | 密码哈希 |
| face_encoding | BLOB | 人脸特征编码（可选） |
| created_at | DATETIME | 创建时间 |
| updated_at | DATETIME | 更新时间 |
| is_active | BOOLEAN | 账户状态 |

### 4.2 Token 黑名单表 (token_blacklist)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PRIMARY KEY | 自增主键 |
| token_jti | VARCHAR(64) UNIQUE | Token 唯一标识 |
| revoked_at | DATETIME | 撤销时间 |

## 5. 接口设计

### 5.1 RESTful API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/auth/register | 用户注册 |
| POST | /api/auth/login | 用户名密码登录 |
| POST | /api/auth/face-login | 人脸识别登录 |
| POST | /api/auth/logout | 登出 |
| POST | /api/auth/refresh | 刷新 Token |
| GET | /api/auth/profile | 获取用户信息 |
| PUT | /api/auth/profile | 更新用户信息 |
| POST | /api/auth/face/register | 注册人脸 |

### 5.2 请求/响应示例

#### 登录请求
```json
POST /api/auth/login
{
  "username": "testuser",
  "password": "SecurePass123!"
}
```

#### 登录响应
```json
{
  "code": 200,
  "message": "登录成功",
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
    "expires_in": 86400
  }
}
```

## 6. 安全设计

### 6.1 密码安全
- 使用 bcrypt 哈希（cost factor = 12）
- 密码强度要求：至少8位，含大小写字母和数字
- 登录失败次数限制（5次/15分钟锁定）

### 6.2 JWT 安全
- 使用 HS256 算法签名
- 密钥从环境变量读取（AUTH_SECRET_KEY）
- Token 包含 jti（唯一标识）用于撤销
- 敏感操作需验证 Token 有效状态

### 6.3 人脸认证安全
- 人脸特征加密存储
- 活体检测建议（眨眼、转头）
- 人脸相似度阈值：0.6（可配置）

### 6.4 其他安全措施
- CORS 跨域限制
- SQL 注入防护（ORM）
- XSS 防护（输入过滤）
- 请求频率限制

## 7. 与现有模块集成

### 7.1 与 AISpeechIntegration 集成
- 登录成功后，语音助手可读取用户偏好
- 支持语音指令控制登录流程

### 7.2 与 FaceRecognitionModule 集成
```python
# face_auth.py 核心逻辑
from FaceRecognitionModule.run_face_recognition import FaceRecognizer

class FaceAuth:
    def __init__(self):
        self.recognizer = FaceRecognizer()
    
    def authenticate(self, face_image):
        """人脸认证"""
        encoding = self.recognizer.encode_face(face_image)
        matched_user = self._match_user(encoding)
        if matched_user:
            return generate_jwt_token(matched_user.id)
        return None
```

## 8. 测试策略

### 8.1 单元测试
- 密码哈希/验证测试
- JWT 生成/验证测试
- 人脸匹配算法测试

### 8.2 集成测试
- 完整登录流程测试
- 人脸识别登录流程测试
- Token 刷新流程测试

### 8.3 安全测试
- SQL 注入测试
- 密码强度边界测试
- Token 过期/篡改测试

## 9. 部署配置

### 9.1 环境变量
```bash
# 认证配置
AUTH_SECRET_KEY=your-secret-key-here
AUTH_TOKEN_EXPIRE_HOURS=24
AUTH_REFRESH_TOKEN_DAYS=7

# 人脸识别配置
FACE_RECOGNITION_THRESHOLD=0.6
FACE_LIVENESS_CHECK=false

# 数据库配置
DATABASE_URL=sqlite:///auth.db
```

### 9.2 依赖安装
```
Flask>=2.3.0
PyJWT>=2.8.0
bcrypt>=4.1.0
SQLAlchemy>=2.0.0
```

## 10. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 人脸识别误识别 | 中 | 设置合理阈值，支持多因素认证 |
| Token 泄露 | 高 | 支持 Token 撤销，HTTPS 传输 |
| 数据库损坏 | 中 | 定期备份，WAL 模式 |

## 11. 里程碑计划

| 阶段 | 内容 | 预估工时 |
|------|------|----------|
| Phase 1 | 用户名密码登录 | 2天 |
| Phase 2 | 人脸识别登录集成 | 2天 |
| Phase 3 | 用户管理与界面 | 1天 |
| Phase 4 | 测试与优化 | 1天 |

---

**文档版本**: v1.0  
**创建日期**: 2026-07-14  
**作者**: System Design Agent
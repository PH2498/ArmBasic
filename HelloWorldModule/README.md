# HelloWorldModule（Helloworld 接口模块）

## 模块职责

提供最简化的 Helloworld 示例 REST 接口，用于演示本项目接口分层与编码规范：
HTTP 协议层（Controller）与业务逻辑层（Service）分离，遵循统一响应结构。

## 关键类说明

| 文件 | 分层 | 职责 |
|------|------|------|
| `helloworld_app.py` | Controller 层 + 启动入口 | 请求分发（路径/方法校验）、统一响应组装、HTTP 服务启动 |
| `helloworld_service.py` | Service 层 | 业务逻辑：生成问候语文本，不依赖 HTTP 细节 |
| `test_helloworld.py` | 单元测试 | 覆盖正常路径、方法校验、路径不存在、JSON 键命名规范 |

## 依赖关系

- 零第三方依赖，仅使用 Python 标准库（`http.server` / `unittest` / `json` / `os` / `sys`）
- 不依赖本项目其他模块

## API 接口列表

| 方法 | 路径 | 说明 | 成功响应 |
|------|------|------|----------|
| GET | `/api/helloworld` | 获取问候语 | 200 |

**请求示例**

```bash
curl http://127.0.0.1:8000/api/helloworld
```

**成功响应**（统一结构，键名 lowerCamelCase）：

```json
{
  "errorCode": "00000",
  "errorMessage": "success",
  "userTip": "",
  "data": {
    "greetMsg": "Hello, World!"
  }
}
```

**错误响应**（包含 HTTP 状态码、errorCode、errorMessage、userTip 四要素）：

```json
{
  "errorCode": "A0001",
  "errorMessage": "method not allowed: only GET is supported",
  "userTip": "仅支持 GET 请求",
  "data": null
}
```

- 非 GET 方法（POST/PUT/DELETE/HEAD）→ `405`
- 未知路径 → `404`

## 运行方式

```bash
# 默认端口 8000
python helloworld_app.py

# 指定端口
python helloworld_app.py --port 8080

# 或通过环境变量
PORT=8080 python helloworld_app.py
```

端口解析优先级：命令行参数 > 环境变量 `PORT` > 默认值 `8000`。

## 测试方式

```bash
cd HelloWorldModule
python3 -m unittest -v
```
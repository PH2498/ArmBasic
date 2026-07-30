# 需求澄清设计文档

- 任务节点：需求澄清
- 采用技能：/brainstorming
- 澄清日期：2026-07-30
- 状态：已按用户指令修正后端落点（D1），静默接管其余决策

---

## 一、需求原始描述

> 用 Java 分别写三个接口 helloworld、哈希算法以及冒泡排序；
> 前端新增一个页面，有三个 tab 分别展示不同的执行结果；
> 新增导出按钮，后台提供导出接口，支持导出各个页面的展示结果；
> 后端再做个埋点，获取调用次数和调用人，前端在当前页面上可视化出来一个报表查看调用情况
> （根据不同的维度：人员类型、人员层级、人员部门等），折线图以及饼图和柱状图不同展示形式。

## 一·补、用户澄清指令

> 后端采用指定仓库开发：ArmBasic

---

## 二、跨仓现状通览（只读探查结论）

| 仓库 | 物理路径 | 性质 | 技术栈 | 与需求相关性 |
|------|---------|------|--------|-------------|
| PH2498.github.io | `…/worktree/PH2498.github.io-master` | 静态博客站 | 纯静态 HTML/CSS/JS（GitHub Pages 风格，含年度归档） | ❌ 不适合 Tab 交互/动态图表 |
| **ArmBasic** ⭐后端 | `…/worktree/ArmBasic-main` | 语音/视觉 AI 项目 | **Python**（dashscope / SpeechRecognition / PyAudio / edge-tts / opencv / face_recognition），入口 `speech_ai.py`、`run_face_recognition.py` | ✅ 用户指定后端落点；但非 Java 工程 |
| iMoney-H5 | `…/worktree/iMoney-H5-main` | 移动端 H5 应用 | umi/max (React) + TS + antd-mobile + **ant-design-mobile-chart** | ✅ 前端落地首选 |
| iMoney | `…/worktree/iMoney-main` | 小程序项目 | project.config.json + src（小程序体系） | ⚠️ 备选，非首选 |

### ArmBasic 仓库结构（确认）
```
ArmBasic/
├─ AISpeechInteraction/   # Python：语音助手（麦克风/ASR/Qwen/TTS）
│  ├─ speech_ai.py
│  ├─ requirements.txt    # dashscope, SpeechRecognition, PyAudio, edge-tts, opencv-python
│  ├─ config_example.env
│  ├─ install_mac.sh
│  └─ static_audio/
├─ FaceRecognitionModule/  # Python：人脸识别（摄像头/face_recognition）
│  ├─ run_face_recognition.py
│  ├─ requirements.txt     # opencv-python, face_recognition, Pillow
│  └─ known_faces/
├─ README.md
└─ .gitignore
```

### 🚨 关键技术栈冲突与处理
- **冲突**：需求要求"用 Java 写后端"，但用户指定的 ArmBasic 是纯 **Python** 项目，无 Java/Maven/Gradle 工程结构与构建链。
- **处理（静默决策）**：尊重用户指定，在 ArmBasic worktree 内**新建独立 Java 后端子模块目录**（如 `ApiServer/`），承载 Spring Boot 服务，与既有 Python 模块目录隔离、并存。Python 模块不被改动（仅新增 Java 模块），满足向后兼容与最小风险。
- **残留风险**：将 Java 工程混入 Python 仓库偏离仓库职责，但用户指令优先；若后续 CI 无 Java 工具链，Java 模块构建需独立环境，不影响 Python 模块运行。

### 已发现有利条件
- iMoney-H5 已内置 `ant-design-mobile-chart`（原生支持折线图/饼图/柱状图，与需求图表形式完全吻合）
- iMoney-H5 已有 `src/pages/Stats` 统计页面（当前为空壳占位 `index.tsx`，可承载埋点报表）
- iMoney-H5 为 umi/max 框架，路由组织规范（`.umirc.ts` routes，已有 Home/AIAssistant/Mine/Stats 路由）
- iMoney-H5 已有 `mock/userAPI.ts`，可作 mock 契约参考

---

## 三、决策点与静默决策

| # | 决策点 | 决策 | 依据 |
|---|--------|------|------|
| D1 | Java 后端代码落点（用户已澄清） | **ArmBasic worktree 内新建 `ApiServer/` 子模块**，Spring Boot 实现，与 Python 模块目录隔离并存 | 用户指令"后端采用 ArmBasic" |
| D2 | 前端落地仓库与形态 | iMoney-H5，新增 `src/pages/Demo` 页面（三 Tab）+ 改造 Stats 页承载埋点报表 | 已有图表库与 Stats 页 |
| D3 | "哈希算法"接口具体实现 | SHA-256，接口入参接收待哈希字符串，返回摘要 | 通用、可演示 |
| D4 | 导出格式 | Excel(.xlsx)，后端用 EasyExcel | 业务常见、结构化 |
| D5 | 埋点用户维度来源 | 后端维护用户体系字典表，按接口调用者身份自动关联人员类型/层级/部门 | 维度字段需稳定来源 |
| D6 | 前端调用后端方式 | umi/max `request`，后端端口与前端 proxy 分离 | 框架约定 |
| D7 | 冒泡排序接口入参 | 数组（JSON），返回升序结果 | 标准语义 |
| D8 | 报表时间维度 | 折线图按"日"维度（近 N 日调用趋势） | 折线天然适配时序 |

---

## 四、接口契约设计（后端，Java/Spring Boot，落点 ArmBasic/ApiServer）

> 仓库落点：`ArmBasic-main/ApiServer/`（建议 `ApiServer/src/main/java/com/mbdemo/apiserver/`）
> 包名前缀 `com.mbdemo.apiserver`（mb = ArmBasic，避免与既有 Python 模块命名冲突）

### 4.1 三个业务接口

| 接口 | Method | Path | 入参 | 出参 |
|------|--------|------|------|------|
| HelloWorld | GET | `/api/hello` | 无 | `{ code:0, data:{ message:"Hello, World!" } }` |
| 哈希算法 | POST | `/api/hash` | `{ algorithm:"SHA-256", input:"<string>" }` | `{ code:0, data:{ algorithm, input, digest } }` |
| 冒泡排序 | POST | `/api/bubble-sort` | `{ numbers:[<int>...] }` | `{ code:0, data:{ original:[...], sorted:[...] } }` |

> 三个接口执行时同步写入埋点记录。

### 4.2 导出接口

| Method | Path | 入参 | 出参 |
|--------|------|------|------|
| GET | `/api/export` | `?type={hello|hash|sort}` | 文件流 `application/vnd.ms-excel`，Content-Disposition: `attachment; filename="<type>-<ts>.xlsx"` |

### 4.3 埋点查询接口

| Method | Path | 入参 | 出参 |
|--------|------|------|------|
| GET | `/api/stats` | `?dimension={type|level|dept}&chart={line|pie|bar}&days={n}` | 聚合数据，结构按 chart 形态适配 |

出参示例（dimension=dept, chart=pie）：
```json
{ "code":0, "data":{ "dimension":"dept", "chart":"pie",
  "items":[ { "name":"研发部", "value":128 }, { "name":"产品部", "value":64 } ] } }
```

出参示例（dimension=type, chart=line, days=7）：
```json
{ "code":0, "data":{ "dimension":"type", "chart":"line",
  "xAxis":["07-24","07-25",...,"07-30"],
  "series":[ { "name":"内部员工", "data":[..] }, { "name":"外部访客", "data":[..] } ] } }
```

### 4.4 埋点记录模型（后端表 `t_api_call_log`）

| 字段 | 说明 |
|------|------|
| id | 主键 |
| api_type | hello / hash / sort |
| caller_id | 调用人标识 |
| caller_type | 人员类型（内部/外部等） |
| caller_level | 人员层级 |
| caller_dept | 人员部门 |
| called_at | 调用时间 |
| cost_ms | 耗时 |

---

## 五、前端设计（iMoney-H5）

### 5.1 路由与页面
- 新增路由 `src/pages/Demo`，三 Tab：`Hello` / `Hash` / `BubbleSort`，各 Tab 触发对应接口并展示执行结果
- 复用/改造 `src/pages/Stats`：承载埋点报表（折线/饼/柱），维度切换 + 图表类型切换
- 导出按钮：Demo 页三 Tab 各配一个"导出当前结果"按钮，调用 `/api/export?type=...`

### 5.2 图表选型（复用 ant-design-mobile-chart）
- 折线图：`Line`（调用趋势，按日）
- 饼图：`Pie`（维度占比）
- 柱状图：`Bar`（维度对比）

### 5.3 数据流
- umi/max `useRequest` / `request` 调后端
- `.umirc.ts` 配 proxy 转发 `/api` → 后端端口

---

## 六、跨库对齐点（契约兼容性）

- **跨库调用链**：前端 iMoney-H5（React）→ HTTP → 后端 ArmBasic/ApiServer（Java/Spring Boot）
- 前后端 JSON 字段命名统一 `camelCase`
- 接口统一响应壳 `{ code, data, message? }`
- 导出 Content-Disposition 文件名前端需解析/或直接走浏览器下载
- 埋点维度枚举前后端一致：`type` / `level` / `dept`
- 图表形态枚举一致：`line` / `pie` / `bar`
- 所有变更向后兼容，仅新增字段/接口，不改既有前端既有路由；后端仅新增 Java 模块，不改 ArmBasic 既有 Python 模块

---

## 七、风险与假设

- 假设：用户体系维度字段（类型/层级/部门）可由后端字典表静态维护；若无真实登录态，后续可降级为 Mock 数据演示
- 假设：ArmBasic worktree 内可引入 Java 构建链（JDK + Maven/Gradle）；若 CI 环境无 Java，Python 模块运行不受影响（ApiServer 子目录独立）
- 风险：在 Python 仓库内混入 Java 后端偏离单仓库职责，但遵循用户指定落点
- 风险：iMoney 小程序仓库 iMoney 与 iMoney-H5 并存，需确认前端是否最终也需覆盖小程序端（本轮默认仅 H5）

---

## 八、产物清点（本轮，需求澄清阶段）

- 本设计文档（仅文档，无代码变更）
- 未修改任何 `.ts` / `.java` / `.py` / `.xml` / `.yaml` 代码文件（符合需求澄清阶段门控）

---

## 九、修订记录

| 版本 | 日期 | 修订内容 |
|------|------|---------|
| v1 | 2026-07-30 | 初版，D1 默认 iMoney-H5/server 子目录（用户拒绝交互澄清） |
| v2 | 2026-07-30 | 用户澄清"后端采用 ArmBasic"，D1 修正为 ArmBasic/ApiServer；补充 ArmBasic 为 Python 仓库的技术栈冲突分析 |

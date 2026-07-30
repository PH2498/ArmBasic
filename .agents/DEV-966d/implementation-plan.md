# API Demo 与埋点报表 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 ArmBasic 仓库内新建 Java/Spring Boot 子模块实现 HelloWorld/哈希/冒泡排序三个接口（含埋点）、导出接口和埋点查询接口，在 iMoney-H5 前端新增 Demo 页（三 Tab + 导出按钮）并改造 Stats 页为埋点可视化报表（折线/饼/柱）。

**Background / Context:**

需求来源：用户要求用 Java 写三个接口（helloworld、哈希算法、冒泡排序），前端新增一个页面有三个 Tab 分别展示执行结果；新增导出按钮，后台提供导出接口支持导出各页面展示结果；后端做埋点获取调用次数和调用人，前端在当前页面可视化报表查看调用情况（按人员类型/层级/部门维度，折线图/饼图/柱状图）。

用户澄清：后端采用指定仓库 ArmBasic。ArmBasic 为 Python 项目（语音/视觉 AI），在 worktree 内新建独立 Java 子模块 `ApiServer/` 与既有 Python 模块隔离并存。

**Current State:**

- `[ArmBasic]` Python 仓库，结构：`AISpeechInteraction/`（speech_ai.py、requirements.txt）、`FaceRecognitionModule/`（run_face_recognition.py）、`README.md`、`.gitignore`。无 Java/Maven/Gradle 工程结构。
- `[iMoney-H5]` umi/max 框架（React 18 + TS），已内置依赖：`antd-mobile`、`ant-design-mobile-chart@^1.2.2`（原生支持 Line/Pie/Bar）、`framer-motion`。现有路由（`.umirc.ts`）：`/home`、`/stats`、`/ai-assistant`、`/mine`。`src/pages/Stats/index.tsx` 为空壳占位页（仅标题"统计"+描述"数据统计页面"）。`src/components/base/` 有 `TabBar.tsx`（底部导航）和 `MotionWrap.tsx`（动画包裹）。`src/constants/index.ts` 有 `TAB_BARS` 导航配置。`mock/userAPI.ts` 有 mock 契约参考（`GET /api/v1/queryUserList`）。`src/models/global.ts` 有全局 model。`src/utils/format.ts` 仅含 trim 工具函数。`.umirc.ts` 已开 `request: {}` 插件但未配 proxy。
- `[iMoney]` 小程序仓库、`[PH2498.github.io]` 静态博客站 → 本轮不涉及。

**Design / Architecture / Key Decisions:**

1. **Java 后端落点**：ArmBasic worktree 内新建 `ApiServer/` 子模块，Spring Boot 3 + Maven，包名 `com.mbdemo.apiserver`，与 Python 模块目录隔离、并存，不动既有 Python 代码。
2. **三个业务接口**：`GET /api/hello`、`POST /api/hash`、`POST /api/bubble-sort`，统一响应壳 `{ code:0, data:{...}, message? }`。
3. **哈希算法**：SHA-256（入参 input 字符串，返回摘要 hex）。
4. **冒泡排序**：入参 numbers 整数数组，返回升序结果 + 原始数组。
5. **导出格式**：Excel(.xlsx)，后端用 EasyExcel，`GET /api/export?type={hello|hash|sort}` 返回文件流。
6. **埋点机制**：三个业务接口执行时同步写入 `t_api_call_log` 表（AOP 或显式调用），记录 api_type/caller_id/caller_type/caller_level/caller_dept/called_at/cost_ms。用户维度由后端用户字典表静态维护（无真实登录态则降级 Mock 数据）。
7. **埋点查询接口**：`GET /api/stats?dimension={type|level|dept}&chart={line|pie|bar}&days={n}`，聚合数据按 chart 形态适配（line → xAxis+series；pie/bar → items[{name,value}]）。
8. **前端 Demo 页**：新增 `src/pages/Demo`，三 Tab（Hello/Hash/BubbleSort），各 Tab 触发对应接口展示结果 + 导出按钮调用 `/api/export?type=...`。
9. **前端报表页**：改造 `src/pages/Stats`，维度切换（type/level/dept）+ 图表类型切换（line/pie/bar），复用 `ant-design-mobile-chart` 的 Line/Pie/Bar。
10. **前端请求层**：新增 `src/services/demo.ts` 封装后端 API 调用；`.umirc.ts` 增 proxy 转发 `/api` → 后端端口（`http://localhost:8080`）。
11. **数据存储**：H2 内存数据库（内嵌、零配置、演示用），Spring Boot 自动建表。生产可替换为 MySQL。
12. **跨库对齐**：前后端 JSON 字段统一 camelCase；维度枚举 `type/level/dept`、图表枚举 `line/pie/bar` 前后端一致；所有变更向后兼容（仅新增）。

**Cross-Repo Interface Contracts (Backend ↔ Frontend):**

| 接口 | Method | Path | 入参 | 出参 |
|------|--------|------|------|------|
| HelloWorld | GET | `/api/hello` | 无 | `{ code:0, data:{ message:"Hello, World!" } }` |
| 哈希算法 | POST | `/api/hash` | `{ algorithm:"SHA-256", input:"<string>" }` | `{ code:0, data:{ algorithm, input, digest } }` |
| 冒泡排序 | POST | `/api/bubble-sort` | `{ numbers:[<int>...] }` | `{ code:0, data:{ original:[...], sorted:[...] } }` |
| 导出 | GET | `/api/export` | `?type={hello\|hash\|sort}` | xlsx 文件流，`Content-Disposition: attachment; filename="<type>-<ts>.xlsx"` |
| 埋点查询 | GET | `/api/stats` | `?dimension={type\|level\|dept}&chart={line\|pie\|bar}&days={n}` | 聚合数据（pie/bar: `{ items:[{name,value}] }`；line: `{ xAxis:[...], series:[{name,data}] }`） |

**Risks & Mitigations:**

- **Java 工具链**：ArmBasic 仓库为 Python 项目，CI 可能无 JDK/Maven。缓解：ApiServer 为独立子目录，构建不依赖 Python 环境；实施阶段用 `mvn` 验证，若环境无 Java 则降级静态审查。
- **仓库职责混合**：Java 模块混入 Python 仓库偏离单仓库职责。缓解：用户指令优先，子模块目录隔离，README 说明。
- **无真实登录态**：用户维度（类型/层级/部门）来源。缓解：后端维护用户字典表静态数据，降级 Mock 演示。
- **H2 演示存储**：重启丢数据。缓解：仅演示用，生产替换 MySQL，配置项隔离。

---

## Task 1: 初始化 ArmBasic/ApiServer Spring Boot 工程骨架

**Repository:** ArmBasic
**Files:**
- `ApiServer/pom.xml` — Maven 构建配置，Spring Boot 3.2.x parent，依赖 spring-boot-starter-web、spring-boot-starter-data-jpa、com.alibaba:easyexcel、com.h2database:h2、lombok
- `ApiServer/src/main/resources/application.yml` — 服务配置，端口 8080，H2 数据源，JPA ddl-auto=update
- `ApiServer/src/main/java/com/mbdemo/apiserver/ApiServerApplication.java` — Spring Boot 主启动类
- `ApiServer/.gitignore` — 忽略 target/

**Implementation Notes:**
- Spring Boot 版本选 3.2.x（需 JDK 17+）
- 包名统一 `com.mbdemo.apiserver`
- H2 配置：`jdbc:h2:mem:apidemo`，用户名 sa，自动建表
- EasyExcel 版本 3.x

**Verification:**
- [ ] `mvn -f ApiServer/pom.xml compile` 成功（若环境有 Java）
- [ ] 主启动类可被 Spring Boot 识别（`@SpringBootApplication`）

---

## Task 2: 后端统一响应壳与请求/响应 DTO

**Repository:** ArmBasic
**Files:**
- `ApiServer/src/main/java/com/mbdemo/apiserver/common/ApiResponse.java` — 统一响应壳 `{ code, data, message }`
- `ApiServer/src/main/java/com/mbdemo/apiserver/dto/HashRequest.java` — `{ algorithm, input }`
- `ApiServer/src/main/java/com/mbdemo/apiserver/dto/HashResponse.java` — `{ algorithm, input, digest }`
- `ApiServer/src/main/java/com/mbdemo/apiserver/dto/BubbleSortRequest.java` — `{ numbers: List<Integer> }`
- `ApiServer/src/main/java/com/mbdemo/apiserver/dto/BubbleSortResponse.java` — `{ original, sorted }`
- `ApiServer/src/main/java/com/mbdemo/apiserver/dto/HelloResponse.java` — `{ message }`

**Implementation Notes:**
- `ApiResponse<T>` 泛型，code=0 表示成功
- 使用 Lombok `@Data` 简化
- 字段命名 camelCase，与前端 TS 对齐

**Verification:**
- [ ] DTO 字段与接口契约表完全一致
- [ ] `mvn compile` 成功

---

## Task 3: 后端埋点数据模型与 Repository

**Repository:** ArmBasic
**Files:**
- `ApiServer/src/main/java/com/mbdemo/apiserver/entity/ApiCallLog.java` — 埋点实体 `t_api_call_log`
- `ApiServer/src/main/java/com/mbdemo/apiserver/repository/ApiCallLogRepository.java` — Spring Data JPA Repository
- `ApiServer/src/main/java/com/mbdemo/apiserver/entity/UserDict.java` — 用户字典实体（caller_id → 类型/层级/部门）
- `ApiServer/src/main/java/com/mbdemo/apiserver/repository/UserDictRepository.java`

**Implementation Notes:**
- `ApiCallLog` 字段：id, apiType, callerId, callerType, callerLevel, callerDept, calledAt, costMs
- `UserDict` 字段：id, callerId, callerType, callerLevel, callerDept（静态维护用户维度）
- JPA ddl-auto=update 自动建表
- Repository 预置聚合查询方法（按维度 group by + count）

**Verification:**
- [ ] 实体字段与需求澄清文档 §4.4 一致
- [ ] Repository 包含按 type/level/dept 维度聚合查询

---

## Task 4: 后端三个业务接口实现（含埋点写入）

**Repository:** ArmBasic
**Files:**
- `ApiServer/src/main/java/com/mbdemo/apiserver/controller/DemoController.java` — 三个接口
- `ApiServer/src/main/java/com/mbdemo/apiserver/service/DemoService.java` — 业务逻辑
- `ApiServer/src/main/java/com/mbdemo/apiserver/service/CallLogService.java` — 埋点写入服务

**Implementation Notes:**
- `GET /api/hello` → 返回 `{ message: "Hello, World!" }`，写埋点 apiType=hello
- `POST /api/hash` → MessageDigest SHA-256，返回 hex digest，写埋点 apiType=hash
- `POST /api/bubble-sort` → 经典冒泡排序（升序），返回 original + sorted，写埋点 apiType=sort
- 埋点写入：记录 callerId（从请求头 X-Caller-Id 或默认 "demo-user"）、calledAt、costMs；caller 维度从 UserDict 查询
- 使用 `System.nanoTime()` 计算 costMs

**Verification:**
- [ ] 三个接口返回结构与契约表一致
- [ ] 每次调用后 t_api_call_log 有对应记录
- [ ] 冒泡排序结果为升序

---

## Task 5: 后端导出接口实现

**Repository:** ArmBasic
**Files:**
- `ApiServer/src/main/java/com/mbdemo/apiserver/controller/ExportController.java`
- `ApiServer/src/main/java/com/mbdemo/apiserver/service/ExportService.java`

**Implementation Notes:**
- `GET /api/export?type={hello|hash|sort}` → EasyExcel 生成 xlsx 文件流
- type=hello：导出 Hello 结果列表（message）
- type=hash：导出哈希历史调用记录（input → digest）
- type=sort：导出排序历史调用记录（original → sorted）
- Content-Disposition: `attachment; filename="<type>-<timestamp>.xlsx"`
- 从 t_api_call_log 查询对应 apiType 的历史数据导出

**Verification:**
- [ ] 三种 type 各返回有效 xlsx 文件流
- [ ] Content-Disposition 正确

---

## Task 6: 后端埋点查询接口实现

**Repository:** ArmBasic
**Files:**
- `ApiServer/src/main/java/com/mbdemo/apiserver/controller/StatsController.java`
- `ApiServer/src/main/java/com/mbdemo/apiserver/service/StatsService.java`
- `ApiServer/src/main/java/com/mbdemo/apiserver/dto/StatsResponse.java`

**Implementation Notes:**
- `GET /api/stats?dimension={type|level|dept}&chart={line|pie|bar}&days={n}`
- pie/bar → `items: [{ name, value }]`（按维度 group by + count）
- line → `xAxis: [日期], series: [{ name, data: [每日count] }]`（按维度 × 日聚合）
- dimension=type → callerType 维度；level → callerLevel；dept → callerDept
- days 默认 7

**Verification:**
- [ ] pie/bar 返回 items 数组
- [ ] line 返回 xAxis + series 结构
- [ ] 维度切换正确聚合对应字段

---

## Task 7: 后端用户字典初始化数据与 Mock 降级

**Repository:** ArmBasic
**Files:**
- `ApiServer/src/main/java/com/mbdemo/apiserver/config/DataInitializer.java` — CommandLineRunner 初始化 UserDict 种子数据
- `ApiServer/src/main/resources/data.sql` — 可选 SQL 种子数据

**Implementation Notes:**
- 预置若干 Mock 用户：覆盖不同 callerType（内部员工/外部访客）、callerLevel（L1/L2/L3）、callerDept（研发部/产品部/运营部）
- 启动时若 UserDict 为空则插入种子数据
- 无真实登录态时 callerId 默认映射到 Mock 用户

**Verification:**
- [ ] 启动后 UserDict 有种子数据
- [ ] 调用接口后埋点记录的 caller 维度非空

---

## Task 8: 前端请求服务层与类型定义

**Repository:** iMoney-H5
**Files:**
- `src/services/demo.ts` — 封装后端 API 调用（hello/hash/bubbleSort/export/stats）
- `src/services/typings.d.ts` — 接口响应类型定义

**Implementation Notes:**
- 使用 umi/max 内置 `request`（`@umijs/max` 导出）
- `helloApi()` → GET /api/hello
- `hashApi(algorithm, input)` → POST /api/hash
- `bubbleSortApi(numbers)` → POST /api/bubble-sort
- `exportApi(type)` → GET /api/export?type=...（responseType: blob，触发浏览器下载）
- `statsApi(dimension, chart, days)` → GET /api/stats
- 类型与后端 DTO 对齐，camelCase

**Verification:**
- [ ] 类型与接口契约一致
- [ ] request 调用路径正确

---

## Task 9: 前端 .umirc.ts 配置 proxy 与路由

**Repository:** iMoney-H5
**Files:**
- `.umirc.ts`（修改）— 新增 proxy 配置 + Demo 路由

**Implementation Notes:**
- 新增 `proxy: { '/api': { target: 'http://localhost:8080', changeOrigin: true } }`
- 新增路由：`{ name: 'Demo', path: '/demo', component: './Demo' }`
- 不改动既有路由（向后兼容）

**Verification:**
- [ ] proxy 配置转发 /api → localhost:8080
- [ ] /demo 路由可访问

---

## Task 10: 前端 Demo 页面 — 三 Tab 与导出按钮

**Repository:** iMoney-H5
**Files:**
- `src/pages/Demo/index.tsx` — Demo 页主组件，三 Tab
- `src/pages/Demo/components/HelloTab.tsx` — Hello Tab
- `src/pages/Demo/components/HashTab.tsx` — Hash Tab
- `src/pages/Demo/components/BubbleSortTab.tsx` — BubbleSort Tab
- `src/pages/Demo/index.less` — 样式

**Implementation Notes:**
- 使用 antd-mobile `Tabs` 组件实现三 Tab（Hello/Hash/BubbleSort）
- HelloTab：按钮触发 GET /api/hello，展示 message
- HashTab：输入框（input + algorithm 选择 SHA-256 默认），按钮触发 POST /api/hash，展示 digest
- BubbleSortTab：输入框（逗号分隔数字），按钮触发 POST /api/bubble-sort，展示 original → sorted
- 每个 Tab 底部配导出按钮，调用 exportApi(type)，blob 下载
- 复用 MotionWrap + TabBar 组件保持页面一致性

**Verification:**
- [ ] 三 Tab 切换正常
- [ ] 各 Tab 调用对应接口展示结果
- [ ] 导出按钮触发文件下载

---

## Task 11: 前端 Stats 报表页改造

**Repository:** iMoney-H5
**Files:**
- `src/pages/Stats/index.tsx`（改造）— 报表主组件
- `src/pages/Stats/components/DimensionSelector.tsx` — 维度切换
- `src/pages/Stats/components/ChartTypeSelector.tsx` — 图表类型切换
- `src/pages/Stats/components/StatsChart.tsx` — 图表渲染（按 chart 类型渲染 Line/Pie/Bar）
- `src/pages/Stats/index.less`（修改）

**Implementation Notes:**
- 维度切换：type / level / dept（antd-mobile Segmented 或 Selector）
- 图表类型切换：line / pie / bar
- 调用 statsApi(dimension, chart, days) 获取数据
- 使用 `ant-design-mobile-chart` 的 Line/Pie/Bar 组件渲染
- line：折线图，xAxis 为日期，series 为多维度数据线
- pie：饼图，items 转为 { name, value } 数据
- bar：柱状图，items 转为 { category, value } 数据
- 切换维度或图表类型时重新请求

**Verification:**
- [ ] 维度切换触发重新查询
- [ ] 图表类型切换渲染对应 Line/Pie/Bar
- [ ] 数据正确展示

---

## Task 12: 前端 constants 新增 Demo 导航入口

**Repository:** iMoney-H5
**Files:**
- `src/constants/index.ts`（修改）— TAB_BARS 新增 demo 项

**Implementation Notes:**
- 新增 `{ key: 'demo', title: 'Demo', icon: 'CodeOutline', path: '/demo' }`
- 需在 @ant-design/icons 导入 CodeOutline
- TabBar.tsx 的 iconMap 新增 CodeOutline 映射
- 注意：若不想占底部 Tab 位，可不加 TAB_BARS 而仅走路由跳转；本计划默认加底部入口便于直达

**Verification:**
- [ ] 底部 TabBar 出现 Demo 入口
- [ ] 点击跳转 /demo

---

## Task 13: TabBar 图标映射补充

**Repository:** iMoney-H5
**Files:**
- `src/components/base/TabBar.tsx`（修改）— iconMap 新增 CodeOutline

**Implementation Notes:**
- `iconMap` 新增 `CodeOutline: { outline: CodeOutlined, filled: CodeFilled }`
- 从 @ant-design/icons 导入 CodeOutlined, CodeFilled
- 与 Task 12 配合

**Verification:**
- [ ] Demo Tab 图标正确显示

---

## Task 14: 端到端联调验证

**Repository:** ArmBasic + iMoney-H5
**Files:** 无新增，验证为主

**Implementation Notes:**
- 启动后端 ApiServer（mvn spring-boot:run，端口 8080）
- 启动前端 iMoney-H5（yarn dev，proxy 转发 /api）
- 验证 Demo 页三 Tab 各接口调用返回正确
- 验证导出按钮下载 xlsx
- 验证 Stats 页维度/图表切换数据正确渲染
- 验证多次调用后埋点数据增长，Stats 反映

**Verification:**
- [ ] 前后端联调三接口正常
- [ ] 导出下载正常
- [ ] 报表数据随调用增长更新
- [ ] 跨库字段契约完全对齐（camelCase、维度枚举、图表枚举）

---

## 跨库对齐点检查清单

- [ ] 统一响应壳 `{ code:0, data:{...}, message? }` 前后端一致
- [ ] JSON 字段全部 camelCase
- [ ] 维度枚举 `type` / `level` / `dept` 前后端一致
- [ ] 图表枚举 `line` / `pie` / `bar` 前后端一致
- [ ] 接口路径 `/api/hello`、`/api/hash`、`/api/bubble-sort`、`/api/export`、`/api/stats` 前后端一致
- [ ] proxy 配置 `/api` → `http://localhost:8080`
- [ ] 后端仅新增 ApiServer 子模块，不改动 ArmBasic 既有 Python 模块（向后兼容）
- [ ] 前端仅新增 Demo 页 + 改造 Stats 页，不改动既有 Home/AIAssistant/Mine 页（向后兼容）

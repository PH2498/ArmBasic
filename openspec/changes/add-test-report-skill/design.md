# Design: add-test-report-skill

> 测试报告生成 Skill 的架构设计。涵盖插件式解析架构、统一数据模型、双模式流程、报告渲染、安全过滤与可维护性扩展点。

## 1. 整体架构

```
┌──────────────────────────────────────────────────────────┐
│                      Skill 入口 (SKILL.md 触发)                        │
│  意图识别 → 模式判定(执行 / 解析) → 配置合并(默认值 + 用户覆盖)          │
└────────────────────────────┬─────────────────────────────┘
                             ▼
        ┌────────────────────┴────────────────────┐
        │ 执行模式: 运行测试命令 → 收集结果产物          │
        │ 解析模式: 跳过执行, 直接读取 result_file      │
        └────────────────────┬────────────────────┘
                             ▼
        ┌────────────────────┴────────────────────┐
        │        解析层 (Parser Registry 插件式)        │
        │  JestParser | VitestParser | PytestParser   │
        │  JUnitXmlParser (跨语言兜底)                  │
        │  → 归一化为统一内部数据模型                    │
        └────────────────────┬────────────────────┘
                             ▼
        ┌────────────────────┴────────────────────┐
        │  覆盖率归集 (可选) → 覆盖率数据模型          │
        │  安全过滤层 (脱敏 + 截断)                    │
        └────────────────────┬────────────────────┘
                             ▼
        ┌────────────────────┴────────────────────┐
        │        报告渲染层 (Renderer 插件式)           │
        │  MarkdownRenderer(默认) | HtmlRenderer(P1)   │
        │  JsonRenderer(P1, 伴随产物)                 │
        │  → 按固定章节顺序渲染 FR2.1-FR2.6             │
        └────────────────────┬────────────────────┘
                             ▼
        ┌────────────────────┴────────────────────┐
        │  落盘 reports/test-report-<ts>.md (FR3.2)     │
        │  返回路径 + 摘要 + 1~3 关键失败原因 (FR3.3)    │
        └─────────────────────────────────────────┘
```

设计要点：
- **解析层与渲染层均为插件式**（NFR5）。新增框架仅新增 Parser，新增格式仅新增 Renderer，互不影响。
- **统一内部数据模型**作为解析层与渲染层间的契约，隔离框架差异（R1）。
- **执行与解析解耦**：解析模式不触碰执行链，天然规避长任务问题（R2）。

## 2. 统一数据模型（归一化内部表示）

所有 Parser 将框架原始产物转换为以下统一模型，渲染层只依赖该模型。

```ts
interface TestReportModel {
  // —— 报告头 (FR2.1) ——
  header: {
    projectName: string;          // 项目名
    generatedAt: string;          // ISO8601 生成时间（幂等性除外字段 NFR4）
    testCommand: string | "未获取"; // 执行命令
    framework: string;            // 框架名，如 jest / vitest / pytest / junit-xml
    frameworkVersion: string | "未获取";
    environment: {                // 执行环境摘要
      node?: string; python?: string; os?: string;
    } | "未获取";
  };

  // —— 结果摘要 (FR2.2) ——
  summary: {
    total: number;
    passed: number;
    failed: number;
    skipped: number;
    passRate: number;            // 0-100
    durationMs: number | "未获取";
    conclusion: "PASS" | "FAIL";  // ✅ / ❌
    meetsThreshold: boolean | null; // fail_threshold 判定，无阈值时 null
  };

  // —— 失败用例分析 (FR2.3) ——
  failures: Array<{
    name: string;                 // 用例名
    file: string | "未获取";       // 所属文件
    errorMessage: string;         // 错误信息
    stackTrace: string[];         // 堆栈关键行（已截断）
  }>;                             // 无失败时为空数组

  // —— 用例明细 (FR2.4) ——
  details: {
    total: number;                // 原始总数
    truncated: boolean;           // 是否超过 200 截断
    shownCount: number;
    groups: Array<{               // 按测试文件分组
      file: string;
      cases: Array<{
        name: string;
        durationMs: number | "未获取";
        status: "passed" | "failed" | "skipped";
      }>;
    }>;
  };

  // —— 覆盖率 (FR2.5) ——
  coverage: {
    available: boolean;
    totals?: {                    // available=true 时存在
      statements: number;         // 0-100
      branches: number;
      functions: number;
      lines: number;
    };
    belowThresholdFiles?: Array<{ // 低于阈值的文件清单
      file: string;
      statements?: number; branches?: number;
      functions?: number; lines?: number;
    }>;
  } | "未获取";                    // 无数据时整体为 "未获取"

  // —— 附录 (FR2.6) ——
  appendix: {
    resultFilePath: string;       // 原始结果文件路径
    toolVersion: string;           // 生成工具版本
  };
}
```

- 缺失字段统一以字符串 `"未获取"` 占位（NFR2 健壮性），渲染层据此标注。
- `generatedAt` 为幂等性例外字段（NFR4），同一结果多次生成报告仅时间戳不同。

## 3. 双模式流程

### 3.1 执行模式（默认）

1. 配置合并：默认值 + 用户覆盖（FR4.2）。
2. 框架/命令识别（FR1.1）：用户指定 > package.json scripts > 框架特征文件。
3. 执行测试命令：
   - 短任务：前台执行，超时阈值由配置控制。
   - 长任务（R2）：交由 Agent 运行时后台任务能力异步执行并轮询，避免阻塞。
4. 收集结果产物：根据框架定位 JSON / XML 结果文件路径。
5. 进入解析层。
6. 执行失败诊断（FR1.4）：命令无法运行时返回诊断信息，不生成空报告。

### 3.2 解析模式

1. 用户指定 `result_file`（如 `./junit.xml`）。
2. 跳过执行，直接读取结果文件。
3. 按 `result_file` 扩展名/内容嗅探选择 Parser（JUnit XML 兜底）。
4. 进入解析层（与执行模式后续流程一致）。
5. 文件损坏（NFR2/AC4）：解析层捕获异常 → 返回明确错误说明，不生成空报告。

## 4. 解析层插件式设计（NFR5）

```
parsers/
  jest.parser.{ts|js}        # Jest JSON → Model
  vitest.parser.{ts|js}      # Vitest JSON → Model
  pytest.parser.{ts|js}      # pytest JUnit XML/JSON → Model (P1)
  junit-xml.parser.{ts|js}   # JUnit XML 兜底 → Model
  registry.{ts|js}           # 注册表: 按框架名/产物特征选择 parser
  base.parser.{ts|js}        # 抽象基类: parse(raw) → TestReportModel
```

- 每个 Parser 实现 `base.parser` 的 `parse(raw: Buffer|string): TestReportModel`。
- `registry` 按优先级选择：用户指定框架 > 特征嗅探（文件内容/扩展名）> JUnit XML 兜底。
- 新增框架（如 Go test / cargo test，M4）：仅新增一个 parser 文件并注册，不动既有代码。

## 5. 渲染层插件式设计

```
renderers/
  markdown.renderer.{ts|js}  # 默认, P0
  html.renderer.{ts|js}      # P1
  json.renderer.{ts|js}      # P1, 结构化伴随产物
  base.renderer.{ts|js}       # 抽象基类: render(model) → string
```

- 每个 Renderer 按 FR2.1-FR2.6 固定章节顺序渲染。
- 章节条件性：失败分析（无失败时省略/标注）、覆盖率（无数据时标注"未获取"）。
- 明细截断（FR2.4）：超 200 条截断并注明，渲染层控制。

## 6. 安全过滤层（NFR3）

在解析层输出 Model 后、渲染层输入前插入安全过滤：

- **凭据脱敏**：对错误信息、堆栈中的环境变量名、`*_TOKEN` / `*_KEY` / `*_SECRET` 模式匹配值替换为 `***`。
- **路径处理**：报告中的文件路径保留相对项目根的路径，不暴露绝对系统路径之外的敏感目录。
- **堆栈截断**：`stackTrace` 截断至可读长度（如最多 5 关键行），超出以 `...(截断 N 行)` 注明。

## 7. 性能考量（NFR1）

- 解析与渲染纯内存操作，1000 用例规模 5 秒内完成。
- 大文件解析采用流式/增量读取，避免全量加载。
- 执行模式的测试耗时不计入 NFR1（仅解析+生成）。

## 8. 幂等性（NFR4）

- 除 `header.generatedAt` 时间戳外，同一结果文件多次生成的报告内容一致。
- 实现保证：Model 的所有字段来源于结果文件，不掺入随机值或当前时间（时间戳除外）。

## 9. 与既有系统的关系

- 本 Skill 为**纯新增**，不修改 `AISpeechInteraction/`、`FaceRecognitionModule/`、`HelloWorld.java` 等既有源码。
- 报告产物落盘到 `reports/`，为运行时产物，不纳入既有源码树（按需 `.gitignore`）。

## 10. 开放设计点（供 openspec-apply 参照）

- Skill 文件具体放置路径：倾向 `.agentix/skills/test-report/`（团队约定路径为准），实施时确认仓库 Skill 目录约定。
- Skill 实现语言：P0 首期目标栈 TypeScript/Node（Q1 决策），解析器与渲染器用 TS 实现。

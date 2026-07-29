# Design: add-test-report-skill

> 本设计面向 `openspec-apply` 阶段实现者，不包含本期实现代码。核心是定义插件式解析器架构、报告生成管道与 Skill 交互约定，使新增框架支持不影响既有解析器（NFR5）。

## 1. 架构总览

Skill 内部采用**三段式管道**，各段之间通过统一归一化数据模型解耦：

```
用户意图 (FR4)
    │
    ▼
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  1. 调度层       │ ─▶ │  2. 解析层（插件式）  │ ─▶ │  3. 报告生成层      │
│  mode/command    │     │  ParserRegistry      │     │  Renderer（MD/HTML）│
│  检测            │     │  → TestRunResult     │     │  → report 文件      │
└─────────────────┘     └──────────────────────┘     └─────────────────────┘
```

- **调度层**：决定执行模式 / 解析模式，检测测试命令与框架，触发执行（必要时后台+轮询）。
- **解析层**：插件式 `Parser`，将各框架原始结果归一化为 `TestRunResult`。
- **报告生成层**：将 `TestRunResult` 喂给 Markdown/HTML/JSON 渲染器，按固定章节输出。

## 2. 统一归一化数据模型 `TestRunResult`

所有解析器的输出契约。解析层差异在此收敛（缓解 R1）。

```ts
interface TestRunResult {
  header: {
    projectName: string;
    generatedAt: string;          // ISO8601，渲染期生成（非解析期），满足 NFR4 幂等
    command: string | "未获取";
    framework: string;            // jest | vitest | pytest | junit | unknown
    frameworkVersion: string | "未获取";
    envSummary: string;           // node/python/os 版本摘要，经脱敏
  };
  summary: {
    total: number;
    passed: number;
    failed: number;
    skipped: number;
    passRate: number;             // 百分比，保留1位小数
    durationMs: number | "未获取";
    overall: "passed" | "failed"; // overall=failed 当 failed>0 或 passRate<fail_threshold
  };
  failures: Array<{
    name: string;                 // 用例名
    filePath: string | "未获取";  // 所属文件
    errorMessage: string;         // 错误信息
    stackKeyLines: string[];      // 堆栈关键行，截断至可读长度（默认≤10行/≤2000字符）
  }>;
  details: {
    groupedByFile: Array<{
      filePath: string;
      cases: Array<{ name: string; durationMs: number | "未获取"; status: "passed"|"failed"|"skipped" }>;
    }>;
    truncated: boolean;          // 超过200条截断
    noteWhenTruncated: string;
  };
  coverage?: {
    statements: number | "未获取";
    branches: number | "未获取";
    functions: number | "未获取";
    lines: number | "未获取";
    belowThresholdFiles: Array<{ filePath: string; metric: string; value: number }>;
  };                              // 不存在时整段省略，章节标注"未获取"（NFR2）
  appendix: {
    sourceResultPath: string | "未获取";
    toolVersion: string;          // 本 Skill 版本
  };
}
```

- 字段缺失统一用字符串 `"未获取"` 或省略可选段，**不得崩溃或静默丢数据**（NFR2）。

## 3. 插件式解析器契约（NFR5）

```ts
interface Parser {
  /** 框架标识，如 "jest" | "vitest" | "junit" | "pytest" */
  framework: string;
  /** 能否处理该输入（按文件特征/内容探测） */
  canHandle(input: ParserInput): boolean;
  /** 解析为 TestRunResult */
  parse(input: ParserInput): TestRunResult;
}

interface ParserInput {
  mode: "execute" | "parse";
  resultFilePath?: string;        // 解析模式必填
  rawOutput?: string;             // 执行模式：命令 stdout/stderr 或落盘产物路径
  coveragePath?: string;         // 可选覆盖率来源
}
```

- `ParserRegistry` 按 `[JUnit兜底]` 优先级最后兜底；任何无法识别的输入回退 JUnit XML 解析器，再不行抛 `UnparsableResultError`（非空报告冒充成功，对应 FR1.4）。
- 新增框架 = 新增一个 Parser 实现 + 注册，不修改既有解析器。

## 4. 报告生成管道

章节顺序固定（FR2），渲染器按 `TestRunResult` 逐段产出：

| 顺序 | 章节 | 数据来源 | 失败用例分析是否必选 |
|---|---|---|---|
| 1 | 报告头 | header | 否（恒定输出） |
| 2 | 结果摘要 | summary（✅/❌ + 量化指标） | 否（恒定输出） |
| 3 | 失败用例分析 | failures | **有失败时必选** |
| 4 | 用例明细 | details（按文件分组，>200截断并注明） | 否（恒定输出） |
| 5 | 覆盖率 | coverage（缺失标注"未获取"，其余章节正常） | 否（条件输出） |
| 6 | 附录 | appendix | 否（恒定输出） |

- Markdown 渲染器为 M1 默认；HTML（M3）、JSON（M3 伴随）为同管道不同 Renderer。
- 幂等：`generatedAt` 与报告内时间戳由渲染期注入；同一 `TestRunResult` 多次渲染，除时间戳外内容一致（NFR4）。

## 5. Skill 交互约定（FR4）

- **触发意图**：`生成测试报告` / `跑一下测试并出报告` / `把这个 junit.xml 转成测试报告`。
- **可配置项**（均有默认值，用户可覆盖）：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| test_command | 自动检测 | 测试执行命令 |
| result_file | 自动检测 | 解析模式下的结果文件路径 |
| output_format | markdown | markdown / html / json |
| output_path | reports/ | 报告输出目录 |
| coverage | auto | auto / on / off |
| fail_threshold | 无 | 通过率低于该值时报告结论标记为不达标 |

- **检测优先级**（FR1.1）：用户显式指定 > 项目配置（package.json scripts.test / pyproject.toml / Cargo.toml）> 框架特征文件（jest.config.* / vitest.config.* / pytest.ini）。
- **生成后返回**（FR3.3）：报告路径 + 结果摘要（通过率、失败数）；失败时附最关键的 1~3 条失败原因。

## 6. 失败诊断与降级（FR1.4 / NFR2）

- 执行模式：命令无法运行（非用例失败）时，返回明确诊断信息（exit code、stderr 摘要、建议），**不得生成空报告冒充成功**。
- 解析模式：结果文件损坏/格式异常时，返回 `UnparsableResultError` 含文件路径与解析失败位置，**非空报告**。
- 字段缺失：降级输出，缺失项标注"未获取"，其余章节正常（NFR2）。
- 用例明细 >200 条：截断并注明截断阈值与被截断总数。

## 7. 安全（NFR3）

- 报告**不得**输出环境变量、密钥、Token。
- 错误堆栈在渲染前经**凭据过滤**：正则脱敏 `Bearer xxx` / `password=` / `token=` / 私钥块等模式。
- `envSummary` 仅保留 node/python/os 版本等非敏感摘要，不拷贝完整 env。

## 8. 性能（NFR1）

- 结果解析 + 报告生成（不含测试执行）1000 用例 ≤ 5 秒。
- 解析器单次遍历产物，不做全量重复 I/O；大报告分段写入，避免内存峰值。

## 9. 目录结构建议（供 openspec-apply 参考）

```
<skill-root>/
  SKILL.md                     # Skill 定义、触发意图、配置项说明
  parsers/
    types.ts                   # Parser / ParserInput / TestRunResult 契约
    jest-parser.*
    vitest-parser.*
    junit-parser.*             # 跨语言兜底
    pytest-parser.*            # M2
    registry.*                 # ParserRegistry + 兜底策略
  renderers/
    markdown-renderer.*        # M1 默认
    html-renderer.*             # M3
    json-renderer.*             # M3 伴随产物
  templates/
    report.zh.md.tmpl          # 中文模板（Q2 决策：仅中文）
  lib/
    detect.*                   # 框架/命令检测
    sanitizer.*                # 凭据脱敏（NFR3）
    config.*                   # 配置项解析与默认值
```

## 10. 与里程碑的映射

| 里程碑 | 设计落点 |
|---|---|
| M1 (P0) | 调度层双模式 + Jest/Vitest/JUnit 解析器 + Markdown 渲染器 + 中文模板 |
| M2 (P1) | pytest 解析器 + 覆盖率章节 + fail_threshold 配置 |
| M3 (P1) | HTML 渲染器 + JSON 伴随产物 |
| M4 (P2 后续) | 历史趋势对比、Go test / cargo test（新解析器，契约不变） |

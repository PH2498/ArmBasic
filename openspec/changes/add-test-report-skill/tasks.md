# Tasks: add-test-report-skill

> 有序、可勾选的实施计划，按里程碑 M1~M3 组织。每个任务可追溯到 spec/design 决策。本期为 propose 阶段，**所有任务未勾选**，交由 `openspec-apply` 执行。

## M1 (P0)：Jest/Vitest + JUnit XML 解析、Markdown 报告、执行/解析双模式

### 基础设施

- [x] 1.1 初始化 Skill 骨架：`SKILL.md`（触发意图、配置项表 FR4.2、与 design.md §5 一致） — 对应 design §5、spec FR4
- [x] 1.2 定义解析器契约与数据模型：`parsers/types.ts`（`Parser` / `ParserInput` / `TestRunResult`，含 `"未获取"` 降级字段） — 对应 design §2/§3、spec NFR2
- [x] 1.3 实现 `ParserRegistry`：按 canHandle 匹配，JUnit 兜底最后，无法解析抛 `UnparsableResultError` — 对应 design §3、spec FR1.4

### 解析器（插件式，NFR5）

- [x] 1.4 实现 `junit-parser`（跨语言兜底，优先支持）：解析 JUnit XML 为 `TestRunResult` — 对应 spec FR1.2 / Scenario 解析模式
- [x] 1.5 实现 `jest-parser`：解析 Jest JSON reporter 产物 — 对应 spec FR1.2 / Scenario 执行模式
- [x] 1.6 实现 `vitest-parser`：解析 Vitest JSON reporter 产物 — 对应 spec FR1.2
- [ ] 1.7 解析器单测：每个解析器对各自样例产物断言 `TestRunResult` 字段正确（含字段缺失降级） — 对应 spec NFR2、design §2

### 调度层（执行/解析双模式 + 框架检测）

- [x] 1.8 实现 `detect`：按 FR1.1 优先级（显式 > package.json scripts.test / pyproject.toml / Cargo.toml > 特征文件）检测命令与框架 — 对应 spec FR1.1 三场景
- [x] 1.9 实现执行模式调度：触发测试命令，长任务交后台+轮询（R2），收集 JSON/XML 产物 — 对应 spec FR1.3 执行模式
- [x] 1.10 实现解析模式调度：不触发执行，直接读 `result_file` — 对应 spec FR1.3 解析模式 / US4
- [x] 1.11 实现命令无法运行的诊断（FR1.4）：非用例失败时返回 exit code/stderr 摘要/建议，不生成空报告 — 对应 spec FR1.4 Scenario

### 报告生成层（Markdown）

- [x] 1.12 实现 Markdown 渲染器：按固定六章节顺序渲染（design §4 表） — 对应 spec FR2 / Scenario 章节顺序
- [x] 1.13 实现失败用例分析章节（有失败必选）：用例名/文件路径/错误信息/堆栈关键行截断 — 对应 spec AC2 / Scenario 失败用例分析
- [x] 1.14 实现用例明细分组与截断（>200 条截断并注明） — 对应 spec FR2 / Scenario 明细截断
- [x] 1.15 实现默认落盘 `reports/test-report-<YYYYMMDD-HHmmss>.md` 与用户指定路径覆盖 — 对应 spec FR3.2 / Scenario 默认落盘
- [x] 1.16 实现生成后返回：报告路径 + 摘要（通过率/失败数）+ 失败时 1~3 条关键原因 — 对应 spec FR3.3

### M1 验证

- [x] 1.17 验证 AC1：在 Jest TS 项目执行"生成测试报告"，产出符合六章节结构的 Markdown，摘要与框架原始输出一致 — 对应 spec AC1
- [x] 1.18 验证 AC2：失败用例报告含用例名/文件路径/错误信息 — 对应 spec AC2
- [x] 1.19 验证 AC3：提供 JUnit XML 走解析模式，不触发执行即产出报告 — 对应 spec AC3 / US4
- [x] 1.20 验证 AC4：损坏结果文件返回明确错误而非空报告 — 对应 spec AC4 / NFR2 Scenario
- [ ] 1.21 验证 NFR1：1000 用例 Jest JSON，解析+生成 ≤5 秒 — 对应 spec NFR1
- [ ] 1.22 验证 NFR4：同一结果多次生成，除时间戳外内容一致 — 对应 spec NFR4

## M2 (P1)：pytest 支持、覆盖率章节、fail_threshold

- [ ] 2.1 实现 `pytest-parser`：解析 pytest JUnit XML/JSON report，不改既有解析器 — 对应 spec NFR5 Scenario
- [ ] 2.2 覆盖率数据接入：`coverage=auto` 探测 Jest/Istanbul/pytest-cov 来源，归一化进 `TestRunResult.coverage` — 对应 design §2 coverage、spec FR2 覆盖率
- [ ] 2.3 覆盖率章节渲染：四指标总表 + 低于阈值文件清单；缺失标注"未获取"且其余章节正常 — 对应 spec AC5 / Scenario 覆盖率缺失
- [ ] 2.4 实现 `fail_threshold`：通过率低于阈值时结论标记不达标 — 对应 spec FR4.2 Scenario fail_threshold
- [ ] 2.5 M2 验证：pytest 项目产出含覆盖率章节的报告；缺失覆盖率场景标注"未获取" — 对应 spec AC5

## M3 (P1)：HTML 输出、JSON 伴随产物

- [ ] 3.1 实现 `html-renderer`：同管道、同六章节顺序，支持 `output_format=html` — 对应 spec FR3.1
- [ ] 3.2 实现 `json-renderer`：输出结构化 `TestRunResult` JSON 作为伴随产物 — 对应 spec FR3.1
- [ ] 3.3 实现 `output_path` 目录自动创建与格式选择落盘 — 对应 spec FR3.2 Scenario 用户指定路径
- [ ] 3.4 M3 验证：HTML 与 JSON 产物章节顺序与 Markdown 一致，幂等（NFR4） — 对应 spec NFR4

## 横切：安全与健壮性（贯穿 M1~M3）

- [ ] X.1 实现凭据脱敏 `sanitizer`：正则过滤 Bearer/password=/token=/私钥块，堆栈与 envSummary 经脱敏 — 对应 spec NFR3 / design §7
- [ ] X.2 验证 NFR3：含凭据的堆栈生成报告后不含原始凭据；envSummary 不含环境变量与密钥 — 对应 spec NFR3 Scenario
- [ ] X.3 验证 NFR2：字段缺失/文件损坏降级输出，缺失项标注"未获取"，不崩溃不静默丢数据 — 对对应 spec NFR2 Scenario

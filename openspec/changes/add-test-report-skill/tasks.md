# Tasks: add-test-report-skill

> 按里程碑 M1→M2→M3 排序的可勾选实施计划。每条任务标注对应 spec 需求与验收标准。M4 为后续迭代，本期仅占位。
>
> **约束**：openspec-propose 阶段不勾选任何任务；不修改产品源码。待 openspec-apply 执行。

## M1: Jest/Vitest JSON + JUnit XML 解析、Markdown 报告、执行/解析双模式 (P0)

### M1-A 基础骨架与配置

- [ ] 确定 Skill 落地路径（倾向 `.agentix/skills/test-report/`，以仓库 Skill 目录约定为准）并创建目录骨架 `[spec: FR4.2]`
- [ ] 编写 `SKILL.md`：触发意图示例（"生成测试报告"/"跑一下测试并出报告"/"把这个 junit.xml 转成测试报告"）、配置项说明与默认值 `[spec: FR4.1, FR4.2]`
- [ ] 实现配置合并模块：默认值 + 用户覆盖（test_command/result_file/output_format/output_path/coverage/fail_threshold） `[spec: FR4.2]`
- [ ] 实现意图识别与模式判定：有 result_file→解析模式，否则执行模式 `[spec: FR1.3, FR4.1]`

### M1-B 统一数据模型

- [ ] 定义 `TestReportModel` 接口（报告头/摘要/失败分析/明细/覆盖率/附录），缺失字段以 `"未获取"` 占位 `[spec: FR2.1-FR2.6, design §2]`
- [ ] 实现 `base.parser` 抽象基类 `parse(raw) → TestReportModel` `[design §4]`

### M1-C 解析器插件（P0 框架）

- [ ] 实现 `JestParser`：Jest JSON → Model，覆盖用例总数/通过/失败/跳过/耗时 `[spec: FR1.2, FR2.2, AC1]`
- [ ] 实现 `VitestParser`：Vitest JSON → Model，字段映射与 JestParser 对齐 `[spec: FR1.2, AC1]`
- [ ] 实现 `JUnitXmlParser`（跨语言兜底）：解析 JUnit XML → Model `[spec: FR1.2, AC3]`
- [ ] 实现 `registry`：按 用户指定框架 > 特征嗅探 > JUnit XML 兜底 选择 parser `[design §4, spec: FR1.2]`
- [ ] 解析层异常处理：文件损坏/格式异常 → 返回明确错误说明，不生成空报告 `[spec: NFR2, FR1.4, AC4]`

### M1-D 执行与收集

- [ ] 实现 FR1.1 框架/命令识别：用户指定 > package.json scripts.test > 框架特征文件推断 `[spec: FR1.1]`
- [ ] 实现执行模式：运行测试命令 → 收集结果产物路径 `[spec: FR1.3]`
- [ ] 执行失败诊断（FR1.4）：命令无法运行时返回诊断信息（命令、错误原因、建议），不冒充成功 `[spec: FR1.4]`
- [ ] 长任务后台执行与轮询（依赖 Agent 运行时后台任务能力），不阻塞主流程 `[design §3.1, risk R2]`

### M1-E 安全过滤层

- [ ] 实现凭据脱敏：`*_TOKEN`/`*_KEY`/`*_SECRET` 模式值替换为 `***` `[spec: NFR3]`
- [ ] 实现堆栈截断：stackTrace 截断至可读长度（如最多 5 关键行）并注明截断 `[spec: FR2.3, NFR3]`

### M1-F Markdown 渲染与落盘

- [ ] 实现 `MarkdownRenderer`：按 FR2.1-FR2.6 固定章节顺序渲染 `[spec: FR2.1-FR2.6, AC1]`
- [ ] 报告头渲染：项目名/生成时间/执行命令/框架版本/环境摘要 `[spec: FR2.1]`
- [ ] 结果摘要渲染：总数/通过/失败/跳过/通过率/耗时/✅❌ 结论 `[spec: FR2.2]`
- [ ] 失败分析章节（有失败时必选）：用例名/文件/错误信息/堆栈关键行 `[spec: FR2.3, AC2]`
- [ ] 用例明细渲染：按文件分组，超 200 条截断并注明 `[spec: FR2.4]`
- [ ] 覆盖率章节：有数据则呈现总表，无数据标注"未获取"且其余章节正常 `[spec: FR2.5, AC5]`
- [ ] 附录渲染：原始结果文件路径/生成工具版本 `[spec: FR2.6]`
- [ ] 实现落盘：默认 `reports/test-report-<YYYYMMDD-HHmmss>.md`，支持用户指定 output_path `[spec: FR3.2]`
- [ ] 实现返回：报告路径 + 摘要（通过率/失败数）+ 1~3 关键失败原因 `[spec: FR3.3, FR2.2]`

### M1-G 测试与验证

- [ ] 构造 Jest JSON 样例，断言 Model 字段与原始输出一致 `[spec: FR2.2, AC1]`
- [ ] 构造含失败用例的样例，断言失败分析含用例名/文件路径/错误信息 `[spec: FR2.3, AC2]`
- [ ] 构造 JUnit XML 样例走解析模式，断言不触发执行即出报告 `[spec: FR1.3, AC3]`
- [ ] 构造损坏结果文件，断言返回明确错误而非空报告 `[spec: NFR2, AC4]`
- [ ] 无覆盖率样例：断言覆盖率标注"未获取"且其余章节正常 `[spec: FR2.5, AC5]`
- [ ] 性能测试：1000 用例样例解析+生成 ≤ 5 秒 `[spec: NFR1]`
- [ ] 幂等性测试：同一结果多次生成，除时间戳外内容一致 `[spec: NFR4]`

## M2: pytest 支持、覆盖率章节、fail_threshold (P1)

- [ ] 实现 `PytestParser`：pytest JUnit XML / JSON report → Model `[spec: FR1.2]`
- [ ] 在 `registry` 注册 PytestParser，支持 pyproject.toml/pytest.ini 推断 `[spec: FR1.1, FR1.2]`
- [ ] 完善覆盖率归集：归一化 statements/branches/functions/lines 四指标 `[spec: FR2.5, design §2]`
- [ ] 覆盖率低于阈值文件清单渲染 `[spec: FR2.5]`
- [ ] `coverage` 配置支持 auto/on/off（auto=有则呈现无则标注未获取） `[spec: FR4.2]`
- [ ] 实现 `fail_threshold`：通过率低于阈值时结论标记不达标 `[spec: FR2.2]`
- [ ] 覆盖率字段缺失降级测试：各指标缺失时标注"未获取" `[spec: NFR2]`

## M3: HTML 输出、JSON 伴随产物 (P1)

- [ ] 实现 `HtmlRenderer`：Model → HTML 报告 `[spec: FR3.1]`
- [ ] 实现 `JsonRenderer`：Model → 结构化 JSON 伴随产物 `[spec: FR3.1]`
- [ ] output_format 配置分发 markdown/html/json `[spec: FR3.1, FR4.2]`
- [ ] HTML/JSON 渲染的一致性测试：与 Markdown 同源 Model 内容一致 `[spec: NFR4]`

## M4: 历史趋势对比、更多框架 (P2，后续迭代，本期占位)

- [ ] [后续迭代] 多次运行结果趋势对比分析 `[proposal: Non-Goals→后续迭代候选]`
- [ ] [后续迭代] `GoTestParser`（Go test） `[spec: FR1.2 扩展, design §4]`
- [ ] [后续迭代] `CargoTestParser`（cargo test） `[spec: FR1.2 扩展, design §4]`
- [ ] [后续迭代] 报告多语言模板（如英文模板），架构已预留扩展点 `[decision: Q2]`

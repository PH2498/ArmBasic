# Proposal: add-test-report-skill

## 概述

提供一个名为 **test-report-skill** 的 Skill，使 Agent 在执行测试后能够自动解析测试结果（执行产物或已有 JUnit XML/JSON），并生成结构化、可读性强的标准测试报告（默认 Markdown）。解决测试结果散落在终端/CI/框架原生输出物中、需人工汇总、跨团队格式不统一、失败上下文需回溯、质量指标无法沉淀的痛点。

本提案为 OpenSpec 提案（clarify 阶段），不包含产品实现代码。批准后由 `openspec-apply` 落地 Skill 本体。

## 澄清决策（Open Questions 已决）

| 开放问题 | 决策 | 依据 |
|---|---|---|
| Q1：首期目标项目栈是否以 TypeScript/Node 为主？ | **是**。P0 范围聚焦 TypeScript/Node（Jest、Vitest）+ 通用 JUnit XML 兜底 | 需求 FR1.2 将 Jest/Vitest 标为 P0；pytest 列入 P1；上下文明确指向 TS/Node 为主 |
| Q2：报告中文/英文双语模板，还是仅中文？ | **仅中文模板** | 需求描述全程中文，US1~US4 均为中文场景；M1 范围无需多语言；双语列为后续迭代候选 |
| Q3：报告是否自动推送 IM/邮件？ | **不做**（维持非目标） | 需求 2.2 明确列为非目标；如需后续由 M4+ 迭代评估 |

## 目标

- **G1**：一条指令（如"生成测试报告"）即可自动完成：执行测试 → 收集结果 → 生成报告。
- **G2**：报告内容标准化，固定顺序包含：报告头、结果摘要、失败用例分析、用例明细、覆盖率、附录 六大板块。
- **G3**：首期支持主流测试框架结果解析：Jest、Vitest（JSON reporter）、JUnit XML（跨语言兜底），M2 增补 pytest。
- **G4**：报告默认 Markdown 输出；HTML（P1）、JSON 伴随产物（P1）作为可选。

## 非目标（本期不做）

- 不做测试用例的自动生成或修复（仅报告）。
- 不做报告的在线托管 / Web 服务化展示。
- 不做多次运行结果的趋势对比分析（后续迭代候选）。
- 不做非测试类质量报告（lint、安全扫描）的聚合。
- 不做 IM / 邮件渠道推送。
- 不做中英双语模板（仅中文）。

## 影响面（Affected Areas）

- **新增**：`openspec/changes/add-test-report-skill/`（本提案文件夹）。
- **后续影响**（由 openspec-apply 落地，本期不实现）：
  - Skill 定义文件（如 `SKILL.md` 及配套脚本/解析器插件目录）。
  - 解析器插件目录结构（Jest/Vitest/JUnit/pytest 解析器）。
  - 报告模板（Markdown 为主，后续 HTML）。
  - 默认输出目录 `reports/`（运行期生成，不入库）。
- **不影响**：现有 `AISpeechInteraction/`、`FaceRecognitionModule/`、`heapsort/` 等业务模块代码。

## 风险与缓解

| 编号 | 风险 | 缓解 |
|---|---|---|
| R1 | 各框架 reporter 输出差异大，解析层抽象复杂 | NFR5 插件式解析器契约；design.md 定义统一归一化数据模型 `TestRunResult` |
| R2 | 测试执行耗时不可控，长任务阻塞 | 执行模式长任务交 Agent 后台执行并轮询；解析模式不触发执行 |
| R3 | 无 OpenSpec CLI，无法自动校验 | 本提案以人工一致性自检替代 `openspec status`；产物遵循标准结构 |
| R4 | 覆盖率来源差异（Jest/Istanbul/pytest-cov） | `coverage=auto` 时按可用性探测，缺失即标注"未获取"，不阻塞报告生成 |

## 回滚（Rollout / Rollback）

- 本提案为纯文档产物，**回滚即删除 `openspec/changes/add-test-report-skill/` 目录**，不影响任何运行代码。
- 若 `openspec-apply` 阶段已落地 Skill 代码，回滚为移除 Skill 文件并清理 `reports/` 运行产物，无数据库/迁移依赖。

## 假设与约束

- 假设：执行环境的 Agent 运行时支持后台任务与轮询（用于长测试任务）。
- 约束：报告生成（不含测试执行本身）在 1000 用例规模下 5 秒内完成（NFR1）。
- 约束：报告中不得泄露环境变量、密钥（NFR3）；错误堆栈须过滤敏感路径外的凭据信息。
- 约束：插件式结构，新增框架支持不影响既有解析器（NFR5）。

## 下一步

批准本提案后，运行 `openspec-apply` 落地 Skill 本体实现。

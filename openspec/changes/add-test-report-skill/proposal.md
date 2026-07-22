# Proposal: add-test-report-skill

> 一键执行测试 → 收集结果 → 生成标准化测试报告的 Skill。

## Intent

当前团队完成测试执行后，测试结果散落在终端输出、CI 日志或框架原生产物（JUnit XML、coverage 目录）中，存在四类痛点：

1. 测试结果需人工收集、整理、汇总，耗时且易遗漏；
2. 缺乏统一格式的测试报告，跨项目/跨团队沟通成本高；
3. 失败用例的上下文（错误信息、堆栈、关联代码）需人工回溯；
4. 覆盖率、通过率等质量指标无法沉淀为可追踪的历史数据。

本提案提供一个 **Skill**，使 Agent 在执行测试后能自动解析测试结果并生成结构化、可读性强的标准测试报告。一条指令（如"生成测试报告"）即可完成：执行测试 → 收集结果 → 生成报告。

## Scope

### 目标（本期必做）

- **G1**：一条指令即可自动完成 执行测试 → 收集结果 → 生成报告；
- **G2**：报告内容标准化，包含 **摘要、明细、失败分析、覆盖率** 四大板块；
- **G3**：支持主流测试框架的结果解析：
  - P0：Jest、Vitest（JSON reporter）；pytest（JUnit XML / JSON report）；JUnit XML（跨语言兜底格式）；
- **G4**：报告支持多种输出格式，默认 Markdown。

### Non-Goals（本期不做，明确标注以防实施越界）

- 不做测试用例的自动生成或修复（仅报告）；
- 不做报告的在线托管 / Web 服务化展示；
- 不做多次运行结果的趋势对比分析（列为后续迭代候选）；
- 不做非测试类质量报告（如 lint、安全扫描）的聚合；
- 不做报告自动推送 IM / 邮件等渠道。

## Affected Areas

| 区域 | 影响说明 |
| --- | --- |
| `openspec/changes/add-test-report-skill/` | 本次提案新增的 OpenSpec change 目录与规划产物 |
| Skill 定义目录（如 `.agentix/skills/test-report/` 或团队约定路径） | 新增 Skill 的 SKILL.md、解析器插件、报告模板；实施阶段由 openspec-apply 落地 |
| 报告输出目录 `reports/` | 运行时产物，默认 `reports/test-report-<YYYYMMDD-HHmmss>.md` |

本提案**不修改既有产品源码**（`AISpeechInteraction/`、`FaceRecognitionModule/`、`HelloWorld.java` 等均不触碰）。

## Key Decisions（开放问题，需求方确认）

> 开放问题 Q1–Q3 在本提案阶段按下述决策落地，实施阶段以此为准。

- **Q1（首期目标项目栈）**：确认 **P0 以 TypeScript/Node 为主栈**（需求文档已据此制定 P0 范围）。Python（pytest）列为 P1，JUnit XML 作为跨语言兜底格式在 P0 即支持。
- **Q2（报告语言模板）**：默认 **中文报告模板**（与团队交互语言一致，安全兜底）。架构上预留多语言模板扩展点，后续可按需增加英文模板，本期不实现双语切换。
- **Q3（是否推送 IM/邮件）**：**不做渠道推送**（需求已明确列为非目标）。仅落盘文件 + 返回摘要给 Agent。

## Risk

| 编号 | 风险 | 缓解策略 |
| --- | --- | --- |
| R1 | 各框架 reporter 输出差异大，解析层易碎 | NFR5 插件式解析器设计，每个框架独立 parser，新增不影响既有（见 design.md） |
| R2 | 测试执行耗时不可控，长任务阻塞 | 执行模式支持后台执行并轮询；依赖 Agent 运行时后台任务能力；解析模式不执行测试，天然规避 |
| R3 | 覆盖率字段在不同工具间命名不一 | 归一化为统一内部数据模型（statements / branches / functions / lines 四指标），缺失项标注"未获取" |
| R4 | 结果文件含敏感信息（路径、凭据） | NFR3 安全过滤：堆栈截断、凭据类内容脱敏，报告不泄露环境变量与密钥 |

## Rollout / Rollback

- **Rollout**：按里程碑 M1（P0）→ M2（P1）→ M3（P1）→ M4（P2 后续）分阶段交付；M1 即可独立可用（Jest/Vitest JSON + JUnit XML 解析、Markdown 报告、执行/解析双模式）。
- **Rollback**：本提案为新增 Skill，不含对既有系统的破坏性变更；如需回滚，移除 Skill 目录与 `reports/` 运行产物即可，不影响项目主代码与既有测试流程。

## Next Step

本提案不含产品代码改动。待用户批准后，下一步执行 **openspec-apply** 按 `tasks.md` 落地实施。

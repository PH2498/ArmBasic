---
name: test-report-skill
description: 解析测试执行结果（Jest/Vitest/pytest/JUnit XML）并生成结构化标准测试报告（默认 Markdown，可选 HTML/JSON）。支持执行模式与解析模式，含失败分析、覆盖率、凭据脱敏。
triggers:
  - 生成测试报告
  - 跑一下测试并出报告
  - 把这个 junit.xml 转成测试报告
  - test report
  - generate test report
version: 1.0.0
language: zh
---

# test-report-skill

> 本 Skill 使 Agent 在执行测试后自动解析测试结果并生成结构化、可读性强的标准测试报告。
> 实现 OpenSpec change `add-test-report-skill`（design.md §9 目录结构）。
> 注：因运行时禁止写入 `skills/` 目录，本实现落盘于工作区根 `test-report-skill/`，目录结构与 design §9 一致。

## 1. 触发意图（FR4.1）

- `生成测试报告`
- `跑一下测试并出报告`
- `把这个 junit.xml 转成测试报告`

命中以上意图即调用本 Skill。意图中若包含结果文件路径（如 `junit.xml`）→ 进入**解析模式**；否则进入**执行模式**。

## 2. 配置项（FR4.2，与 design.md §5 一致）

均有默认值，用户可覆盖：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `test_command` | 自动检测 | 测试执行命令 |
| `result_file` | 自动检测 | 解析模式下的结果文件路径 |
| `output_format` | `markdown` | `markdown` / `html` / `json` |
| `output_path` | `reports/` | 报告输出目录 |
| `coverage` | `auto` | `auto` / `on` / `off` |
| `fail_threshold` | 无 | 通过率低于该值时报告结论标记为不达标（百分比，如 `80`） |

## 3. 工作流

1. **解析配置**：读取用户覆盖项，与默认值合并（`lib/config.mjs`）。
2. **检测框架/命令**：按 FR1.1 优先级（显式 > 项目配置 > 特征文件），`lib/detect.mjs`。
3. **选择模式**：
   - 执行模式：触发 `test_command`（长任务后台+轮询），收集产物。
   - 解析模式：直接读 `result_file`，不触发执行（US4）。
4. **解析结果**：`ParserRegistry` 按插件匹配，JUnit 兜底；无法解析抛 `UnparsableResultError`（FR1.4，非空报告）。
5. **渲染报告**：按六章节固定顺序渲染（Markdown 默认；HTML/JSON 可选）。
6. **落盘**：默认 `reports/test-report-<YYYYMMDD-HHmmss>.md`，允许用户指定。
7. **返回**：报告路径 + 摘要（通过率/失败数）+ 失败时 1~3 条关键原因。

## 4. 报告章节（FR2，顺序固定）

1. 报告头：项目名、生成时间、执行命令、框架/版本、执行环境摘要
2. 结果摘要：用例总数、通过/失败/跳过、通过率、总耗时、✅/❌ 结论
3. 失败用例分析（有失败时必选）：用例名、所属文件、错误信息、堆栈关键行（截断）
4. 用例明细：按测试文件分组，>200 条截断并注明
5. 覆盖率（若可获取）：四指标总表 + 低于阈值文件清单；缺失标注"未获取"
6. 附录：原始结果文件路径、生成工具版本

## 5. 失败诊断与降级（FR1.4 / NFR2）

- 命令无法运行（非用例失败）：返回 exit code、stderr 摘要、建议，**不生成空报告冒充成功**。
- 结果文件损坏：返回 `UnparsableResultError`（含文件路径与解析失败位置），**非空报告**。
- 字段缺失：降级输出，缺失项标注"未获取"，其余章节正常。

## 6. 安全（NFR3）

- 报告不得输出环境变量、密钥、Token。
- 错误堆栈渲染前经凭据脱敏（`lib/sanitizer.mjs`）：`Bearer xxx` / `password=` / `token=` / 私钥块。
- `envSummary` 仅保留 node/python/os 版本摘要，不拷贝完整 env。

## 7. 文件清单

```
test-report-skill/
  SKILL.md
  parsers/
    types.mjs          # Parser / ParserInput / TestRunResult / UnparsableResultError
    registry.mjs      # ParserRegistry（JUnit 兜底）
    junit-parser.mjs  # 跨语言兜底
    jest-parser.mjs
    vitest-parser.mjs
    pytest-parser.mjs # M2
  renderers/
    markdown-renderer.mjs
    html-renderer.mjs # M3
    json-renderer.mjs # M3
  templates/
    report.zh.md.tmpl
  lib/
    detect.mjs        # 框架/命令检测
    sanitizer.mjs     # 凭据脱敏
    config.mjs        # 配置默认值
    junit-xml.mjs     # 轻量 JUnit XML 解析（无外部依赖）
    env.mjs           # 环境摘要（脱敏）
  index.mjs           # 调度层 + CLI 入口
  test/
    fixtures/         # 样例产物
    run-tests.mjs     # 单测/验证脚本
```

## 8. CLI 用法（供 Agent 经 exec 调用）

```bash
# 执行模式：自动检测框架并跑测试
node test-report-skill/index.mjs

# 解析模式：直接解析已有结果文件
node test-report-skill/index.mjs --result-file reports/junit.xml

# 指定输出格式与路径
node test-report-skill/index.mjs --result-file reports/junit.xml --format html --output reports/
```

支持配置项以 `--<key> <value>` 覆盖。

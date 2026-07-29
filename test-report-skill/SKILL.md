---
name: test-report-skill
description: >-
  Parse test results (Jest/Vitest JSON, pytest JUnit XML/JSON, generic JUnit XML)
  and generate a structured, readable standard test report (default Markdown).
  One command ("生成测试报告") completes: execute tests → collect results → generate report.
  Supports both execute-mode and parse-mode (reuse existing result files, e.g. JUnit XML).
triggers:
  - 生成测试报告
  - 跑一下测试并出报告
  - 把这个 junit.xml 转成测试报告
config:
  test_command:
    default: auto
    description: 测试执行命令；auto 表示自动检测
  result_file:
    default: auto
    description: 解析模式下的结果文件路径；auto 表示自动检测
  output_format:
    default: markdown
    description: markdown / html / json
  output_path:
    default: reports/
    description: 报告输出目录
  coverage:
    default: auto
    description: auto / on / off
  fail_threshold:
    default: 无
    description: 通过率低于该值(%)时报告结论标记为不达标
---

# Test Report Generation Skill

执行测试后自动解析结果（Jest/Vitest JSON、pytest JUnit XML/JSON、通用 JUnit XML）并生成结构化、可读性强的标准测试报告（默认 Markdown）。

## 触发意图

- 生成测试报告
- 跑一下测试并出报告
- 把这个 junit.xml 转成测试报告

## 两种模式

- **执行模式**：Skill 触发测试运行并收集结果
- **解析模式**：跳过执行，直接解析用户指定的已有结果文件（满足 US4 / CI 复用）

当 `result_file` 显式指定且 `test_command` 为 auto 时自动进入解析模式。

## 可配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| test_command | auto | 测试执行命令 |
| result_file | auto | 解析模式下的结果文件路径 |
| output_format | markdown | markdown / html / json |
| output_path | reports/ | 报告输出目录 |
| coverage | auto | auto / on / off |
| fail_threshold | 无 | 通过率低于该值时报告结论标记为不达标 |

## 报告结构（顺序固定）

1. 报告头
2. 结果摘要
3. 失败用例分析（仅有失败时渲染）
4. 用例明细
5. 覆盖率
6. 附录

## 支持框架（P0）

- JavaScript/TypeScript：Jest、Vitest（JSON reporter）
- Python：pytest（JUnit XML / JSON report）
- 通用：JUnit XML（跨语言兜底格式）

## 性能与限制

- 结果解析与报告生成 ≤ 5s（1000 用例规模，NFR1）
- 用例明细 >200 条时截断并注明（FR2.4）
- 敏感数据（环境变量、密钥）在报告前过滤（NFR3）

## 扩展指南（NFR5 插件式）

新增框架支持三步：
1. 实现 `ParserPlugin` 接口（`canHandle` / `parse`）
2. 注册到 `defaultRegistry`
3. 添加 fixture 测试

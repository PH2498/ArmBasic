# Spec: test-report-skill

> 测试报告生成 Skill 的用户可见需求与场景。每个需求均有可测的 Given/When/Then 场景，供 openspec-apply 实施时验收。

## FR1: 测试执行与结果收集

### FR1.1 框架与命令自动识别

- **Given** 项目中存在 `package.json` 且其 `scripts.test` 定义了测试命令（如 `jest`）
- **When** 用户执行"生成测试报告"且未显式指定 `test_command`
- **Then** Skill 以 `package.json` 的 `scripts.test` 作为测试命令

- **Given** 项目中存在 `jest.config.js` 或 `vitest.config.ts` 等框架特征文件，但 `package.json` 无 `scripts.test`
- **When** 用户执行"生成测试报告"且未显式指定命令
- **Then** Skill 按框架特征文件推断对应框架（jest→jest、vitest→vitest）

- **Given** Python 项目存在 `pyproject.toml` / `pytest.ini`
- **When** 用户执行"生成测试报告"且未指定命令
- **Then** Skill 推断为 pytest（P1 里程碑落地）

- **Given** 用户显式指定 `test_command`（如 `npm run test:ci`）
- **When** 执行"生成测试报告"
- **Then** 用户指定命令优先级最高，覆盖自动检测结果

### FR1.2 框架与结果格式支持

- **Given** 项目使用 Jest，且测试命令配置了 JSON reporter
- **When** 生成报告
- **Then** Skill 解析 Jest JSON 结果产物（P0）

- **Given** 项目使用 Vitest，配置了 JSON reporter
- **When** 生成报告
- **Then** Skill 解析 Vitest JSON 结果产物（P0）

- **Given** 项目使用 pytest，产出 JUnit XML 或 JSON report
- **When** 生成报告
- **Then** Skill 解析对应格式（P1）

- **Given** 项目使用任意框架但产出 JUnit XML
- **When** 生成报告
- **Then** Skill 以 JUnit XML 作为跨语言兜底格式解析（P0）

### FR1.3 执行模式与解析模式

- **Given** 默认执行模式（未指定 `result_file`）
- **When** 用户执行"生成测试报告"
- **Then** Skill 触发测试运行并收集结果，再生成报告

- **Given** 解析模式（用户指定 `result_file` 指向已有结果文件，如 `./junit.xml`）
- **When** 用户执行"把这个 junit.xml 转成测试报告"
- **Then** Skill 跳过测试执行，直接解析指定文件生成报告（满足 US4 CI 复用场景）

### FR1.4 执行失败诊断

- **Given** 测试命令无法运行（如命令不存在、依赖缺失、非用例失败的命令级错误）
- **When** 生成报告
- **Then** Skill 返回明确诊断信息（命令、错误原因、建议），**不得生成空报告冒充成功**

---

## FR2: 报告内容（标准结构）

报告章节顺序固定，缺一不可（除标注条件性外）。

### FR2.1 报告头

- **Given** 已成功收集测试结果
- **When** 生成报告
- **Then** 报告头包含：项目名、生成时间、执行命令、框架/版本、执行环境摘要

### FR2.2 结果摘要

- **Given** 已解析结果数据
- **When** 生成报告
- **Then** 结果摘要包含：用例总数、通过数、失败数、跳过数、通过率、总耗时；整体结论用 ✅ / ❌ 标识

- **Given** 配置了 `fail_threshold`（通过率阈值）
- **When** 实际通过率低于阈值
- **Then** 报告结论标记为"不达标"（❌ 或明确文字）

### FR2.3 失败用例分析（有失败时必选）

- **Given** 存在失败用例
- **When** 生成报告
- **Then** 失败分析章节包含每条失败用例：用例名、所属文件、错误信息、堆栈关键行（截断至可读长度）

- **Given** 无失败用例
- **When** 生成报告
- **Then** 失败分析章节省略或明确标注"无失败用例"

### FR2.4 用例明细

- **Given** 已解析结果数据
- **When** 生成报告
- **Then** 用例明细按测试文件分组，列出用例名与各自耗时

- **Given** 用例总数超过 200 条
- **When** 生成报告
- **Then** 明细截断展示前 200 条并注明截断（默认展示全部，超过 200 条时截断并注明）

### FR2.5 覆盖率（若可获取）

- **Given** 覆盖率数据存在
- **When** 生成报告
- **Then** 覆盖率章节含 语句/分支/函数/行 覆盖率总表，以及低于阈值的文件清单

- **Given** 覆盖率数据不存在
- **When** 生成报告
- **Then** 覆盖率章节标注"未获取"，且其余章节正常呈现（满足 AC5）

### FR2.6 附录

- **Given** 报告生成完成
- **When** 生成报告
- **Then** 附录含：原始结果文件路径、生成工具版本

---

## FR3: 输出格式与落盘

### FR3.1 输出格式

- **Given** 默认配置 `output_format=markdown`
- **When** 生成报告
- **Then** 产出 `.md` 文件（P0）

- **Given** `output_format=html`
- **When** 生成报告
- **Then** 产出 `.html` 文件（P1）

- **Given** `output_format=json`
- **When** 生成报告
- **Then** 产出结构化 JSON 作为伴随产物（P1）

### FR3.2 默认输出路径

- **Given** 用户未指定 `output_path`
- **When** 生成报告
- **Then** 报告落盘到 `reports/test-report-<YYYYMMDD-HHmmss>.md`

- **Given** 用户指定 `output_path`（如 `./out/report.md`）
- **When** 生成报告
- **Then** 报告落盘到用户指定路径

### FR3.3 生成后返回

- **Given** 报告生成成功
- **When** 返回给用户
- **Then** 返回：报告路径 + 结果摘要（通过率、失败数）

- **Given** 报告生成成功且存在失败用例
- **When** 返回给用户
- **Then** 附最关键的 1~3 条失败原因

---

## FR4: Skill 交互约定

### FR4.1 触发意图

- **Given** 用户说"生成测试报告" / "跑一下测试并出报告" / "把这个 junit.xml 转成测试报告"
- **When** Skill 识别意图
- **Then** 触发对应工作流（执行模式或解析模式）

### FR4.2 可配置项

- **Given** 配置项 `test_command`（默认自动检测）、`result_file`（默认自动检测）、`output_format`（默认 markdown）、`output_path`（默认 `reports/`）、`coverage`（默认 auto）、`fail_threshold`（默认无）
- **When** 用户未覆盖某项
- **Then** 使用默认值；用户可覆盖任一项

---

## NFR1: 性能

- **Given** 1000 用例规模的结果文件
- **When** 执行结果解析与报告生成（不含测试执行本身）
- **Then** 完成时间 ≤ 5 秒

## NFR2: 健壮性

- **Given** 结果文件格式异常或字段缺失
- **When** 解析与生成报告
- **Then** 降级输出，缺失项标注"未获取"，不得崩溃或静默丢数据（满足 AC4）

## NFR3: 安全

- **Given** 结果文件含环境变量、密钥类内容或敏感路径
- **When** 生成报告
- **Then** 报告不泄露环境变量与密钥；错误堆栈过滤敏感凭据信息

## NFR4: 幂等性

- **Given** 同一结果文件
- **When** 多次生成报告
- **Then** 内容一致（时间戳字段除外）

## NFR5: 可维护性

- **Given** 新增框架支持需求（如 Go test / cargo test）
- **When** 扩展解析器
- **Then** 以插件式新增 parser，不影响既有解析器代码

---

## 验收标准映射（Acceptance Criteria）

| AC | 对应场景 | 里程碑 |
| --- | --- | --- |
| AC1 | FR2.1-FR2.4（TS 项目 Jest/Vitest，Markdown 报告结构完整，摘要与原始输出一致） | M1 |
| AC2 | FR2.3（失败分析含用例名、文件路径、错误信息） | M1 |
| AC3 | FR1.3（JUnit XML 解析模式不触发执行） | M1 |
| AC4 | NFR2/FR1.4（结果文件损坏返回明确错误而非空报告） | M1 |
| AC5 | FR2.5（覆盖率存在则呈现，不存在标注"未获取"且其余正常） | M1/M2 |

## 用户故事映射（User Stories）

| US | 对应需求 |
| --- | --- |
| US1 开发者本地生成报告 | FR1.1/FR1.3 执行模式 + FR3.2 落盘 + FR3.3 返回路径 |
| US2 失败用例定位 | FR2.3 失败分析（错误信息、堆栈、源文件路径） |
| US3 QA 同步质量状态 | FR2.2 摘要可量化指标（通过率、覆盖率） |
| US4 CI 复用解析模式 | FR1.3 解析模式不重复跑测试 |

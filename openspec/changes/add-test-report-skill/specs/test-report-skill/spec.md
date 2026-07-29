# Spec: test-report-skill

> Status: ADDED。本 spec 定义 test-report-skill 的用户可见需求与验收场景，覆盖 FR1~FR4、AC1~AC5、NFR1~NFR5。所有场景采用 Given/When/Then 格式，保证可测。

## ADDED Requirements

### Requirement: 测试执行与结果收集（FR1）

Skill 应支持自动识别测试框架与运行命令，首期支持 Jest/Vitest（JSON reporter）、JUnit XML（跨语言兜底），并支持执行/解析双模式。执行失败（非用例失败）须给出明确诊断，不得生成空报告冒充成功。

#### Scenario: 自动检测测试命令（FR1.1，优先级 a>b>c）

- **Given** 一个含 `package.json`（scripts.test 存在）且无 `jest.config.*` 特征冲突的 TS/Node 项目
- **When** 用户未显式指定 test_command，触发"生成测试报告"
- **Then** Skill 优先采用 `package.json` 的 `scripts.test` 作为执行命令，而非框架特征文件推断

#### Scenario: 显式指定命令优先级最高（FR1.1 a）

- **Given** 用户在触发时显式指定 `test_command=npm run test:ci`
- **When** 执行测试
- **Then** 使用该命令执行，忽略项目配置与特征文件推断

#### Scenario: 执行模式触发测试并收集结果（FR1.3 执行模式）

- **Given** 一个 Jest 项目，已产出 JSON reporter 结果路径约定
- **When** 用户触发"跑一下测试并出报告"（执行模式）
- **Then** Skill 运行测试命令，收集 JSON 产物并解析为 `TestRunResult`

#### Scenario: 解析模式跳过执行（FR1.3 解析模式，US4）

- **Given** 用户指向一个已存在的 `junit.xml` 文件并触发"把这个 junit.xml 转成测试报告"
- **When** 解析模式启动
- **Then** Skill **不触发任何测试执行**，直接解析该文件产出报告

#### Scenario: 命令无法运行时拒绝空报告（FR1.4）

- **Given** 测试命令因依赖缺失无法运行（exit code 非 0 且非用例失败）
- **When** 执行模式尝试运行
- **Then** Skill 返回明确诊断信息（exit code、stderr 摘要、建议），**不生成空报告**，不标记为成功

### Requirement: 报告内容标准结构（FR2）

生成的报告必须按固定顺序包含六大章节：报告头、结果摘要、失败用例分析（有失败时必选）、用例明细、覆盖率（若可获取）、附录。

#### Scenario: 报告章节顺序固定（FR2）

- **Given** 任意一份有效 `TestRunResult`
- **When** 生成 Markdown 报告
- **Then** 章节依次为：① 报告头 ② 结果摘要 ③ 失败用例分析（仅当 failures 非空） ④ 用例明细 ⑤ 覆盖率（可获取时） ⑥ 附录

#### Scenario: 结果摘要含量化指标与结论标识（FR2 / US3）

- **Given** 一份含 total=10、passed=8、failed=2、skipped=0 的结果
- **When** 生成报告
- **Then** 结果摘要含用例总数、通过/失败/跳过数、通过率(80.0%)、总耗时，整体结论以 ❌ 标识（因 failed>0）

#### Scenario: 失败用例分析含定位信息（FR2 / AC2 / US2）

- **Given** 一条失败用例，含用例名、所属文件、错误信息、堆栈
- **When** 生成报告
- **Then** 失败用例分析章节包含该用例的用例名、文件路径、错误信息、堆栈关键行（截断至可读长度）

#### Scenario: 用例明细超 200 条截断并注明（FR2）

- **Given** 一份含 250 条用例明细的结果
- **When** 生成报告
- **Then** 用例明细按测试文件分组展示，超过 200 条时截断并注明截断阈值与被截断总数

#### Scenario: 覆盖率缺失时不阻塞其余章节（FR2 / AC5 / NFR2）

- **Given** 一份无覆盖率数据的结果
- **When** 生成报告
- **Then** 覆盖率章节标注"未获取"，其余章节（报告头/摘要/失败分析/明细/附录）正常输出

### Requirement: 输出格式与落盘（FR3）

默认 Markdown 输出，落盘至 `reports/test-report-<YYYYMMDD-HHmmss>.md`，允许用户指定路径；生成后返回报告路径与结果摘要。

#### Scenario: 默认 Markdown 落盘且路径明确（FR3.2 / FR3.3 / US1 / AC1）

- **Given** 用户未指定 output_path
- **When** 生成报告
- **Then** 报告以 `.md` 写入 `reports/test-report-<YYYYMMDD-HHmmss>.md`，并向用户返回该路径 + 摘要（通过率、失败数）

#### Scenario: 用户指定输出路径（FR3.2）

- **Given** 用户指定 `output_path=./out/` 且 `output_format=html`
- **When** 生成报告
- **Then** 报告以 `.html` 写入用户指定目录

#### Scenario: 失败时附关键失败原因（FR3.3）

- **Given** 存在失败用例
- **When** 报告生成后向用户返回摘要
- **Then** 附最关键的 1~3 条失败原因（按失败用例顺序取前 1~3 条）

### Requirement: Skill 交互约定（FR4）

支持自然语言触发与可配置项覆盖（test_command / result_file / output_format / output_path / coverage / fail_threshold），均有默认值。

#### Scenario: 自然语言触发意图（FR4.1）

- **Given** 用户输入"生成测试报告"
- **When** Skill 匹配触发意图
- **Then** 进入默认执行模式（自动检测命令与格式）

#### Scenario: 配置项覆盖默认值（FR4.2）

- **Given** 用户指定 `coverage=off`
- **When** 生成报告
- **Then** 即使存在覆盖率数据，报告也不呈现覆盖率章节（或标注关闭），其余配置项保持默认

#### Scenario: fail_threshold 触发不达标结论（FR4.2）

- **Given** `fail_threshold=90`，实际通过率 80%
- **When** 生成报告
- **Then** 报告结论标记为不达标（❌）

### Requirement: 非功能需求（NFR）

性能、健壮性、安全、幂等性、可维护性约束。

#### Scenario: 解析与生成性能达标（NFR1）

- **Given** 一份 1000 用例规模的 Jest JSON 结果
- **When** 执行解析 + 报告生成（不含测试执行本身）
- **Then** 全程在 5 秒内完成

#### Scenario: 字段缺失降级不崩溃（NFR2 / AC4）

- **Given** 一份损坏或字段缺失的结果文件
- **When** 解析
- **Then** Skill 返回明确错误说明（含文件路径与解析失败位置），不崩溃；可降级部分输出时缺失项标注"未获取"，不静默丢数据

#### Scenario: 报告不泄露凭据（NFR3）

- **Given** 一条失败用例的错误堆栈中含 `Bearer eyJxxx` / `password=secret` / `token=abc`
- **When** 生成报告
- **Then** 堆栈关键行经脱敏后呈现，不含原始凭据；envSummary 不含环境变量与密钥

#### Scenario: 幂等生成（NFR4）

- **Given** 同一结果文件
- **When** 多次生成报告
- **Then** 除时间戳字段外，报告内容一致

#### Scenario: 插件式新增框架不影响既有解析器（NFR5）

- **Given** 已有 Jest/Vitest/JUnit 解析器
- **When** 新增 pytest 解析器
- **Then** 仅新增一个 Parser 实现 + 注册，不修改既有解析器源码，既有解析器行为不变

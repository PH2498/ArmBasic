# 代码评审报告 — QuickSort

> 评审技能：`/dtazziboot-java-code-review`（SDD 范式结构化审查）
> 评审阶段：review / 代码评审
> 评审日期：2026-07-27
> 评审对象：快速排序算法实现（需求：实现一个快速排序算法）

---

## 一、执行队列（Step1）

| # | 文件 | 类型 | 角色 |
|---|------|------|------|
| 1 | `QuickSort/pom.xml` | 构建 | Maven 工程描述 / 依赖 / 插件 |
| 2 | `QuickSort/src/main/java/com/shuke/algo/sort/QuickSort.java` | 主代码 | 快速排序算法工具类（被测对象） |
| 3 | `QuickSort/src/test/java/com/shuke/algo/sort/QuickSortTest.java` | 测试 | 单元测试（JUnit 5） |

---

## 二、预扫结果（scan-all-rules.sh）

[降级说明] 预扫脚本 `references/script/scan-all-rules.sh` 存在且可执行，但本次运行其 stdout 被运行时以敏感输出拦截、且 `cd /workspace` 目标路径不存在，未能捕获脚本扫描明细。依据防超时降级协议，已切换为「静态代码审查」，覆盖原预扫对应的逻辑分支、边界条件、类型一致性与可靠性规则（B/M/I/A/S/G 各类规则的人工等价核查见下文）。已审查的逻辑点：null 安全、空/单元素边界、分区正确性、递归终止、重复元素、原地引用、不可变副本、大数交叉验证。

---

## 三、逐文件审查

### 3.1 `QuickSort/pom.xml`

**功能核对**
- `groupId=com.shuke.algo`、`artifactId=quick-sort`、`version=1.0.0-SNAPSHOT`、`packaging=jar`，与工程目录 `QuickSort` 一致。
- 编译目标 `source/target=8`，与主代码（无 Java 8 以上语法）匹配。
- 测试依赖 `junit-jupiter 5.10.2`（test scope），与测试类 `@Test`（Jupiter）一致。
- 插件 `maven-compiler-plugin 3.13.0`、`maven-surefire-plugin 3.2.5`，版本集中到 `<properties>`，可维护。

**可读性 / 可靠性 / 扩展**
- 版本号、编码 `UTF-8`、sourceEncoding 均显式声明，符合规范。
- 无快照依赖、无 `<repositories>` 私服地址硬编码、无 SNAPSHOT 传递依赖风险。
- 结论：**通过**，无问题。

---

### 3.2 `QuickSort/src/main/java/com/shuke/algo/sort/QuickSort.java`

#### 功能核对
- 需求为"实现快速排序算法"。实现采用 **Lomuto 分区方案**，取区间末位元素为基准，升序、原地排序，逻辑正确：
  - `partition`：`pivot=array[high]`，`i=low-1`，`j` 遍历 `[low, high)`，`array[j]<=pivot` 时 `i++` 并 `swap(i,j)`，最后 `swap(i+1,high)` 返回 `i+1`。标准且正确。
  - `quickSort`：`low>=high` 终止，先分区再左右递归。终止条件正确，不会越界。
  - `sort`：`null` 或 `length<=1` 直接返回，随后 `quickSort(array,0,length-1)`，返回同一引用。与 Javadoc 契约一致。
  - `sortedCopy`：`null` 返回 `null`；否则 `Arrays.copyOf` 后排序，不改原数组。契约一致。
- 公开 API 行为与 Javadoc 完全自洽，功能达标。

#### 可读性检查
- 类/方法/常量均有 Javadoc，含复杂度说明（平均 O(n log n)、最坏 O(n²)、空间 O(log n)），信息充分。
- 命名清晰：`quickSort` / `partition` / `swap` / `sortedCopy`。
- 工具类私有构造 + 注释，符合惯例。

**问题 CR-R02 [P2/Info，可读性]**：常量 `PIVOT_TAIL_OFFSET=0` 恒为 0 且 `private static final` 不可变，却以 `array[high + PIVOT_TAIL_OFFSET]` 形式使用，引入无意义的间接层，使"取末位基准"这一直观操作被掩盖。建议直接写 `array[high]`；若要让基准选择策略真正可配置，应抽象为策略接口而非一个固定 0 的偏移量。

**问题 CR-R03 [P2/Info，可读性]**：`MIN_PARTITION_LENGTH=1` 在 `sort()` 中作为"数组长度阈值"使用，而名称语义偏"分区长度"，两者概念略有错位（此处实际语义为"长度≤1 直接返回"）。命名可更精确，如 `MIN_SORTABLE_LENGTH`。

#### 可靠性检查
- **null 安全**：`sort`/`sortedCopy` 均先判 `Objects.isNull`，无 NPE 风险。
- **边界**：空数组（`length=0`→`<=1` 命中返回）、单元素（`length=1`→返回）、两元素均覆盖；`quickSort` 的 `low>=high` 兜底保证不会对 0/1 长度区间误递归。
- **越界**：`partition` 中 `j` 上界为 `high`（不含），`i+1` 落在 `[low, high]`，`swap(i+1, high)` 合法；`swap` 内 `i==j` 提前返回避免自交换无意义写。无 `ArrayIndexOutOfBounds`。
- **类型一致性**：全程 `int[]`/`int`，无装箱/拆箱、无类型混淆。
- **线程安全**：纯静态、无共享可变状态，线程安全。

**问题 CR-R01 [P1/Major，可靠性]**：Lomuto + 末位基准在**已升序/已降序/大量重复**输入下退化为最坏情况，时间 O(n²)、递归深度 O(n)。对于大体积有序输入存在 `StackOverflowError` 风险（Javadoc 已声明最坏复杂度，但未声明栈溢出风险）。当前测试最大规模 1000 且为伪随机分布，未触发；生产环境若处理大体积有序数据将暴露该风险。建议二选一或多选：
  1. 基准策略升级为三数取中（median-of-three）或随机化基准，规避最坏分区；
  2. 小区间切换插入排序（如 `high-low < 16` 时走插入排序），降低递归深度与常数；
  3. 对较长一侧改迭代、短侧递归（尾递归消除），将栈深控制在 O(log n)。

#### 自定义扩展检查（对照数科 Java 编码规范方向）
- 魔法常量已提取为命名常量（方向正确，但见 CR-R02/R03 的过设计点）。
- Javadoc 含 `@author`/`@date`，符合规范。
- 测试方法命名 `sort_xxx_shouldYyy` 见下节，风格统一。
- 无 `System.out`、无吞异常、无裸 `catch`、无未关闭资源。

---

### 3.3 `QuickSort/src/test/java/com/shuke/algo/sort/QuickSortTest.java`

#### 功能核对
- 覆盖矩阵完备：乱序、已升序、倒序、含重复、单元素、空数组、`null`、原地同引用、`sortedCopy` 不改原数组、`sortedCopy(null)`、大数交叉验证（对齐 `Arrays.sort`）、全相同、负正混合。共 13 个用例，与主代码公开 API 契约一一对应。
- 断言选用恰当：`assertArrayEquals` 校内容、`assertSame` 校原地引用、`assertNull` 校 null 契约、`assertNotNull` 校空数组非 null。
- `sort_largeRandom_shouldMatchJdkSort` 以 JDK 内置排序做交叉验证，是良好实践。

#### 可读性 / 可靠性
- 类/方法 Javadoc 齐全，显式声明遵循 FIRST 原则。
- 用例独立、无共享状态、自校验、可重复。
- `sort_largeRandom` 使用确定性伪随机 `(i*73+37)%997`，保证可重复（符合 FIRST 的 Repeatable），是合理取舍。

**问题 CR-T01 [P2/Info，测试]**：`sort_largeRandom` 的伪随机模式虽可重复，但分布偏置、且未覆盖 CR-R01 所述的**最坏情况输入（大体积已有序数组）**，无法对 P1 栈深风险形成回归防护。建议补一个 `sort_largeSorted_shouldNotStackOverflow`（如 50000 升序）用例，作为 CR-R01 修复后的回归锚点；在 CR-R01 未修复前可标记 `@Disabled` 并注明原因，避免 CI 不稳定。

**问题 CR-T02 [P2/Info，测试]**：缺少 `sortedCopy` 对**空数组**与**单元素**的契约用例（`sort` 有，`sortedCopy` 仅覆盖了乱序与 null）。建议补 `sortedCopy_emptyArray` / `sortedCopy_singleElement`，使不可变副本路径的边界与 `sort` 对齐。

---

## 四、问题汇总

| 编号 | 等级 | 类别 | 位置 | 摘要 |
|------|------|------|------|------|
| CR-R01 | P1/Major | 可靠性 | QuickSort.java L68-76,88-89 | 末位基准致最坏 O(n²)/O(n) 递归，大体积有序输入有 StackOverflow 风险 |
| CR-R02 | P2/Info | 可读性 | QuickSort.java L20,89 | `PIVOT_TAIL_OFFSET=0` 恒定常量引入无意义间接，建议直接 `array[high]` |
| CR-R03 | P2/Info | 可读性 | QuickSort.java L25,41 | `MIN_PARTITION_LENGTH` 命名与"数组长度阈值"语义错位 |
| CR-T01 | P2/Info | 测试 | QuickSortTest.java L127-137 | 大数用例未覆盖最坏情况有序输入，缺 CR-R01 回归锚点 |
| CR-T02 | P2/Info | 测试 | QuickSortTest.java | `sortedCopy` 缺空数组/单元素边界用例 |

> 无 P0/Blocker。P1 1 项（CR-R01），P2 4 项。

---

## 五、评审结论

**结论：有条件通过（Conditional Pass）**

- 功能正确性、可读性、测试覆盖度整体达到交付水准，无 P0 阻断项。
- CR-R01（P1）为已知算法特性且已在 Javadoc 声明复杂度，但栈溢出风险未声明且无回归用例；**建议在合入前或紧随合后的迭代中处理**（基准随机化 / 小区间插入排序 / 尾递归消除，任一即可显著缓解）。
- CR-R02/R03/T01/T02 均为 P2 改进项，不阻断合入，可在后续小步重构中收敛。

### 建议处理顺序
1. CR-R01：升级基准策略 + 补 CR-T01 回归用例（高优先）。
2. CR-T01/CR-T02：补齐测试边界（随 CR-R01 一并提交）。
3. CR-R02/R03：常量与命名清理（低优先，可独立提交）。

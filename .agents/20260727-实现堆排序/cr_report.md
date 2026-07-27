# Code Review Report

> **Change** `实现堆排序` · **分支/Commit** `AI/task-DEV-966dcd0a-7905-11f1-9649-3b4281182f10-603d84ce-cb08-4a06-a740-b7a6a0ffb05e` / `344bfe1` · **日期** `2026-07-27` · **审查者** AI
>
> **AI**：等级 **P0 / P1 / P2**；G/S 以 checklist 行内定义为准；Bug 模式以 `bug-pattern-checklist.md` 表头为准（Blocker→P0、Major→P1、Info→P2）。已运行 `scan-all-rules.sh` 并将要点并入 §5，再写 LLM 结论。问题含 `path:line` 或清单 ID。

---

## 1. 审查范围

| 项 | 值 |
|----|-----|
| `.java` 文件数 | `2` |
| 变更行数 | `+409 / -0`（含 pom.xml +45） |

| 类/接口 | 路径 | 角色（可选） |
|---------|------|--------------|
| `HeapSort` | `heapsort/src/main/java/com/dtazziboot/algorithm/sort/HeapSort.java` | 堆排序算法门面（153 行） |
| `HeapSortTest` | `heapsort/src/test/java/com/dtazziboot/algorithm/sort/HeapSortTest.java` | 单元测试（211 行） |
| `pom` | `heapsort/pom.xml` | 构建配置（JDK 8 + JUnit 4.13.2） |

---

## 2. 问题计数

| P0 | P1 | P2 |
|----|----|-----|
| 0 | 0 | 1 |

---

## 3. Step 2 — 功能（REQ）

> 依据 `design.md`：F05 边界与参数校验、F06 用例覆盖、H01-R05 不越界访问右子、H05-R03 相等不交换避免死循环。需求：实现一个堆排序。

### REQ-1: 自然序升序排序（大顶堆，原地）

| Scenario | 结果 | Spec证据 | 代码证据 | 说明 |
|----------|------|----------|----------|------|
| 常规乱序 | ✅ | `design.md` §算法流程：建堆→交换堆顶至末尾→缩堆→siftDown | `HeapSort.java:79-87`（doSort）、`:96-101`（buildHeap） | 建堆自 `n/2-1` 向前批量下沉，复杂度 O(n)；交换+siftDown 主循环 O(n log n)，符合设计 |
| 已排序/逆序/重复/全等 | ✅ | F06 用例覆盖 | `HeapSortTest.java:40-72` | 测试覆盖最好/最坏/重复/全等场景 |
| 相等元素不死循环 | ✅ | H05-R03：cmp==0 不交换 | `HeapSort.java:131` `cmp.compare(arr[largest], arr[parent]) > 0` | 仅 `>0` 才下沉，`==0` 走 break 分支，无死循环风险 |

### REQ-2: 自定义比较器排序（方向由比较器决定）

| Scenario | 结果 | Spec证据 | 代码证据 | 说明 |
|----------|------|----------|----------|------|
| 降序（reverseOrder） | ✅ | `design.md` §接口：方向由比较器决定 | `HeapSort.java:59-70`、测试 `HeapSortTest.java:125-130` | reverseOrder 下数值大者 compare 为负，数值小者浮至堆顶→交换到末尾，得降序，静态推演与测试一致 |
| 自定义对象按属性 | ✅ | F06 用例覆盖 | `HeapSortTest.java:134-153`（Person by age） | `Comparator.comparingInt` 正确驱动 |
| 不越界访问右子 | ✅ | H01-R05 | `HeapSort.java:127` `right < end && ...` | 短路求值，仅左子存在时不访问右子 |

### REQ-3: 边界与参数校验（F05）

| Scenario | 结果 | Spec证据 | 代码证据 | 说明 |
|----------|------|----------|----------|------|
| null 数组 | ✅ | F05：抛 IllegalArgumentException | `HeapSort.java:40-42`、`:60-62` | 两个重载均校验 |
| null 比较器 | ✅ | F05 | `HeapSort.java:63-65` | 比较器重载单独校验 |
| 空数组/单元素 | ✅ | F05：静默返回 | `HeapSort.java:43-45`、`:66-68` `length < 2` 短路 | 避免无意义建堆 |

**功能结论**：需求"实现一个堆排序"完整覆盖，升序/降序/比较器/边界/异常均符合系分设计。

---

## 4. Step 3 — 可读性检查

| 结果 | 说明（违规写 Ax.x 与 `path:行`） |
|------|--------------------------------|
| ✅ | 无可读性违规。类为 `final` + 私有构造（`HeapSort.java:24,27-28`）符合工具类规范；方法命名 `sort/doSort/buildHeap/siftDown/swap` 语义清晰；类/方法均有 Javadoc，含时间/空间复杂度与线程安全说明（`:5-23,30-38,49-58`）；`2*parent+1` 等索引公式有行内注释（`:106,119,126`）。 |

---

## 5. Step 4 — 可靠性检查

> **scan-all-rules.sh 预扫**（52/222 条已扫，输出原文）：
> ```
> [P0] G16.2 — CatchWithoutLogging: heapsort/src/test/.../HeapSortTest.java:96
> [P0] G16.2 — CatchWithoutLogging: heapsort/src/test/.../HeapSortTest.java:107
> [P0] G16.2 — CatchWithoutLogging: heapsort/src/test/.../HeapSortTest.java:118
> Summary: 3 findings (P0=3, P1=0, P2=0)
> ```
> **LLM 复核**：上述 3 处均为测试文件的 `try { ... fail() } catch (IllegalArgumentException e) { assertEquals(...) }` 模式（`:92-99,103-110,113-121`），属 JUnit 标准异常断言，目的是验证异常类型与消息，非生产代码吞异常。G16.2 规则针对"捕获异常不记录日志导致问题被掩盖"，对测试断言属**误报**，降级为 N/A，不计入 P0。主源码 `HeapSort.java` 无任何脚本命中。

| 域 | 参考 | 结果 | 等级 | 说明（列命中 ID 或「已扫无命中」） |
|----|------|------|------|-------------------------------------|
| 可靠性 | `reliability-checklist.md` G1 并发 | N/A | — | 纯算法库，无共享状态；Javadoc 明示并发由调用方加锁（`HeapSort.java:17`） |
| 可靠性 | G2 幂等 / G3 事务 | N/A | — | 非写接口/消息消费，无事务 |
| 可靠性 | G4–G15（IO/资源/限流/熔断等） | N/A | — | 纯内存计算，无外部依赖 |
| 可靠性 | G16 异常处理 | ✅ | — | 主源码无 catch，采用 fast-fail throw（`:41,61,64`）；测试 catch 经复核为误报 |
| 可靠性 | 边界条件 | ✅ | — | 空/单/null 已覆盖（见 §3 REQ-3） |
| Bug 模式 | `bug-pattern-checklist.md` B/M/I | ✅ | — | 已扫无命中：B002 ArrayEquals（测试用 `assertArrayEquals` 正确）、B004 ArrayToString 无命中、其余无 |
| 安全 | `security-checklist.md` S1–S10 | N/A | — | 纯算法库，无 SQL/认证/CSRF/密钥/外部输入 |

### §5.1 P2 建议（Info，非阻断）

| ID | 等级 | 位置 | 说明 |
|----|------|------|------|
| 索引溢出理论隐患 | P2 | `HeapSort.java:119` `int left = 2 * parent + 1` | 当 `parent` 接近 `Integer.MAX_VALUE/2` 时 `2*parent` 可能溢出为负致 `left<0`。实际受 JVM 数组长度上限（远小于该阈值）约束不会触发，仅作健壮性提示，可不修改。 |

---

## 6. 结论

- **Blocker（P0）数：0**
- 实现严格遵循 `design.md` 系分设计：大顶堆升序 + 比较器方向 + 原地排序 + O(1) 额外空间（迭代 siftDown）。
- `scan-all-rules.sh` 预扫的 3 处 G16.2 命中经 LLM 复核为测试断言误报，已降级，不计入 Blocker。
- 测试覆盖正常/边界/异常/比较器场景，21 个用例，静态推演算法正确性通过。
- **评审通过，可进入下一阶段。**

---

## 7. 问题片段（附 `.java` 代码）

> 本变更无 P0/P1 问题。G16.2 三处命中（误报）相关片段如下，供复核留痕：

### §7.1 G16.2 误报片段（测试异常断言，已降级 N/A）

`heapsort/src/test/java/com/dtazziboot/algorithm/sort/HeapSortTest.java:92-99`

```java
@Test
public void testSortNullArrayThrows() {
    try {
        HeapSort.sort((Integer[]) null);
        fail("expected IllegalArgumentException for null array");
    } catch (IllegalArgumentException e) {
        assertEquals("array must not be null", e.getMessage());
    }
}
```

**复核说明**：`catch` 块用于断言异常类型与消息，配合 `fail()` 确保抛出路径命中，非生产代码吞异常，G16.2 不适用。

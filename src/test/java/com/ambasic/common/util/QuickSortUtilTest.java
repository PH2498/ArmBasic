package com.ambasic.common.util;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Comparator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link QuickSortUtil} 单元测试。
 *
 * <p>遵循 FIRST 原则，覆盖正常路径、边界值、异常路径与分支。
 * 测试类与被测类位于同一包路径，便于访问包级 API。</p>
 *
 * @author ArmBasic
 * @since 1.0.0
 */
@DisplayName("QuickSortUtil 快速排序工具类测试")
class QuickSortUtilTest {

    /** 整型排序测试用例共享的已排序期望数组（升序）。 */
    private static final int[] EXPECTED_ASC = {1, 2, 3, 4, 5};

    /** 空数组常量，用于边界测试。 */
    private static final int[] EMPTY_ARRAY = {};

    /** 单元素数组常量，用于边界测试。 */
    private static final int[] SINGLE_ELEMENT = {42};

    /** 单元素期望数组常量。 */
    private static final int[] SINGLE_EXPECTED = {42};

    @Nested
    @DisplayName("整型数组全量排序 sort(int[])")
    class IntArraySortTest {

        @Test
        @DisplayName("正常路径：乱序数组升序排序")
        void shouldSortUnsortedArray() {
            // Arrange
            int[] array = {3, 1, 4, 1, 5, 2};

            // Act
            QuickSortUtil.sort(array);

            // Assert
            assertThat(array).isSorted();
        }

        @Test
        @DisplayName("边界值：空数组排序后仍为空")
        void shouldHandleEmptyArray() {
            // Arrange
            int[] array = EMPTY_ARRAY;

            // Act
            QuickSortUtil.sort(array);

            // Assert
            assertThat(array).isEmpty();
        }

        @Test
        @DisplayName("边界值：单元素数组排序后保持不变")
        void shouldHandleSingleElement() {
            // Arrange
            int[] array = SINGLE_ELEMENT;

            // Act
            QuickSortUtil.sort(array);

            // Assert
            assertThat(array).isEqualTo(SINGLE_EXPECTED);
        }

        @Test
        @DisplayName("分支：已排序数组排序后保持不变")
        void shouldHandleAlreadySortedArray() {
            // Arrange
            int[] array = {1, 2, 3, 4, 5};

            // Act
            QuickSortUtil.sort(array);

            // Assert
            assertThat(array).isEqualTo(EXPECTED_ASC);
        }

        @Test
        @DisplayName("分支：逆序数组排序后变为升序")
        void shouldSortReverseOrder() {
            // Arrange
            int[] array = {5, 4, 3, 2, 1};

            // Act
            QuickSortUtil.sort(array);

            // Assert
            assertThat(array).isEqualTo(EXPECTED_ASC);
        }

        @Test
        @DisplayName("分支：含重复元素的数组排序后保持稳定升序")
        void shouldSortWithDuplicates() {
            // Arrange
            int[] array = {3, 1, 2, 1, 3, 2};

            // Act
            QuickSortUtil.sort(array);

            // Assert
            assertThat(array).containsExactly(1, 1, 2, 2, 3, 3);
        }

        @Test
        @DisplayName("分支：含负数的数组排序后为升序")
        void shouldSortWithNegativeNumbers() {
            // Arrange
            int[] array = {0, -1, 5, -10, 3};

            // Act
            QuickSortUtil.sort(array);

            // Assert
            assertThat(array).containsExactly(-10, -1, 0, 3, 5);
        }

        @Test
        @DisplayName("异常路径：null 数组抛出 IllegalArgumentException")
        void shouldThrowWhenArrayIsNull() {
            // Arrange
            int[] array = null;

            // Act & Assert
            assertThatThrownBy(() -> QuickSortUtil.sort(array))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("不能为 null");
        }
    }

    @Nested
    @DisplayName("整型数组区间排序 sort(int[], int, int)")
    class IntArrayRangeSortTest {

        @Test
        @DisplayName("正常路径：对部分区间排序，区间外元素不受影响")
        void shouldSortSpecifiedRangeOnly() {
            // Arrange
            int[] array = {9, 3, 1, 4, 5, 0};

            // Act
            QuickSortUtil.sort(array, 1, 4);

            // Assert
            assertThat(Arrays.copyOfRange(array, 1, 5)).containsExactly(1, 3, 4, 5);
            assertThat(array[0]).isEqualTo(9);
            assertThat(array[5]).isEqualTo(0);
        }

        @Test
        @DisplayName("异常路径：startIndex 大于 endIndex 抛出 IllegalArgumentException")
        void shouldThrowWhenStartIndexGreaterThanEndIndex() {
            // Arrange
            int[] array = {1, 2, 3};

            // Act & Assert
            assertThatThrownBy(() -> QuickSortUtil.sort(array, 2, 1))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("区间非法");
        }

        @Test
        @DisplayName("异常路径：startIndex 为负数抛出 IllegalArgumentException")
        void shouldThrowWhenStartIndexIsNegative() {
            // Arrange
            int[] array = {1, 2, 3};

            // Act & Assert
            assertThatThrownBy(() -> QuickSortUtil.sort(array, -1, 1))
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("异常路径：endIndex 超出数组长度抛出 IllegalArgumentException")
        void shouldThrowWhenEndIndexOutOfBound() {
            // Arrange
            int[] array = {1, 2, 3};

            // Act & Assert
            assertThatThrownBy(() -> QuickSortUtil.sort(array, 0, 5))
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("异常路径：null 数组抛出 IllegalArgumentException")
        void shouldThrowWhenArrayIsNull() {
            // Arrange
            int[] array = null;

            // Act & Assert
            assertThatThrownBy(() -> QuickSortUtil.sort(array, 0, 1))
                    .isInstanceOf(IllegalArgumentException.class);
        }
    }

    @Nested
    @DisplayName("泛型数组排序 sort(T[], Comparator)")
    class GenericArraySortTest {

        @Test
        @DisplayName("正常路径：字符串数组按自然顺序升序排序")
        void shouldSortStringArrayAscending() {
            // Arrange
            String[] array = {"banana", "apple", "cherry"};

            // Act
            QuickSortUtil.sort(array, Comparator.naturalOrder());

            // Assert
            assertThat(array).containsExactly("apple", "banana", "cherry");
        }

        @Test
        @DisplayName("正常路径：整数对象数组按降序排序")
        void shouldSortIntegerArrayDescending() {
            // Arrange
            Integer[] array = {3, 1, 4, 1, 5};

            // Act
            QuickSortUtil.sort(array, Comparator.reverseOrder());

            // Assert
            assertThat(array).containsExactly(5, 4, 3, 1, 1);
        }

        @Test
        @DisplayName("边界值：空数组排序后仍为空")
        void shouldHandleEmptyArray() {
            // Arrange
            String[] array = new String[0];

            // Act
            QuickSortUtil.sort(array, Comparator.naturalOrder());

            // Assert
            assertThat(array).isEmpty();
        }

        @Test
        @DisplayName("异常路径：null 数组抛出 IllegalArgumentException")
        void shouldThrowWhenArrayIsNull() {
            // Arrange
            String[] array = null;

            // Act & Assert
            assertThatThrownBy(() -> QuickSortUtil.sort(array, Comparator.naturalOrder()))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("不能为 null");
        }

        @Test
        @DisplayName("异常路径：null 比较器抛出 IllegalArgumentException")
        void shouldThrowWhenComparatorIsNull() {
            // Arrange
            String[] array = {"a", "b"};
            Comparator<String> comparator = null;

            // Act & Assert
            assertThatThrownBy(() -> QuickSortUtil.sort(array, comparator))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("比较器不能为 null");
        }
    }

    @Nested
    @DisplayName("副本排序 sortedCopy(int[])")
    class SortedCopyTest {

        @Test
        @DisplayName("正常路径：返回排序后的新数组，原数组不变")
        void shouldReturnSortedCopyWithoutModifyingOriginal() {
            // Arrange
            int[] array = {3, 1, 2};

            // Act
            int[] sorted = QuickSortUtil.sortedCopy(array);

            // Assert
            assertThat(sorted).containsExactly(1, 2, 3);
            assertThat(array).containsExactly(3, 1, 2);
        }

        @Test
        @DisplayName("边界值：空数组返回空副本")
        void shouldReturnEmptyCopyForEmptyArray() {
            // Arrange
            int[] array = EMPTY_ARRAY;

            // Act
            int[] sorted = QuickSortUtil.sortedCopy(array);

            // Assert
            assertThat(sorted).isEmpty();
            assertThat(sorted).isNotSameAs(array);
        }

        @Test
        @DisplayName("异常路径：null 数组抛出 IllegalArgumentException")
        void shouldThrowWhenArrayIsNull() {
            // Arrange
            int[] array = null;

            // Act & Assert
            assertThatThrownBy(() -> QuickSortUtil.sortedCopy(array))
                    .isInstanceOf(IllegalArgumentException.class);
        }
    }
}

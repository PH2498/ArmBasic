package com.dtazziboot.algorithm.sort;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * {@link QuickSorter} 单元测试。
 *
 * <p>覆盖设计文档业务规则 R01~R04 及边界场景：null 入参、空/单元素短路、
 * 重复元素稳定性、已排序/逆序/随机数据正确性、null 元素抛 NPE。</p>
 */
class QuickSorterTest {

    /**
     * 边界输入场景。
     */
    @Nested
    @DisplayName("边界输入")
    class BoundaryInputTest {

        @Test
        @DisplayName("R01: 入参为 null 时抛出 NullPointerException")
        void shouldThrowNpeWhenArrayIsNull() {
            assertThrows(NullPointerException.class, () -> QuickSorter.sort(null));
        }

        @Test
        @DisplayName("空数组短路返回，不抛异常")
        void shouldShortCircuitOnEmptyArray() {
            Integer[] array = new Integer[0];
            assertDoesNotThrow(() -> QuickSorter.sort(array));
            assertArrayEquals(new Integer[0], array);
        }

        @Test
        @DisplayName("单元素数组短路返回，内容不变")
        void shouldShortCircuitOnSingleElementArray() {
            Integer[] array = {42};
            QuickSorter.sort(array);
            assertArrayEquals(new Integer[]{42}, array);
        }
    }

    /**
     * 正确性场景。
     */
    @Nested
    @DisplayName("排序正确性")
    class CorrectnessTest {

        @Test
        @DisplayName("随机乱序整数数组排序后升序")
        void shouldSortRandomArray() {
            Integer[] array = {5, 2, 9, 1, 5, 6};
            Integer[] expected = {1, 2, 5, 5, 6, 9};
            QuickSorter.sort(array);
            assertArrayEquals(expected, array);
        }

        @Test
        @DisplayName("已排序数组排序后保持不变")
        void shouldSortAlreadySortedArray() {
            Integer[] array = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            Integer[] expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            QuickSorter.sort(array);
            assertArrayEquals(expected, array);
        }

        @Test
        @DisplayName("逆序数组排序后升序")
        void shouldSortReverseOrderedArray() {
            Integer[] array = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
            Integer[] expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            QuickSorter.sort(array);
            assertArrayEquals(expected, array);
        }

        @Test
        @DisplayName("全相同元素数组排序后保持不变")
        void shouldSortAllEqualArray() {
            Integer[] array = {7, 7, 7, 7, 7, 7};
            Integer[] expected = {7, 7, 7, 7, 7, 7};
            QuickSorter.sort(array);
            assertArrayEquals(expected, array);
        }

        @Test
        @DisplayName("含重复元素数组排序后升序且重复元素相邻")
        void shouldSortArrayWithDuplicates() {
            Integer[] array = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
            Integer[] expected = {1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9};
            QuickSorter.sort(array);
            assertArrayEquals(expected, array);
        }

        @Test
        @DisplayName("两元素无序数组排序后升序")
        void shouldSortTwoElementUnorderedArray() {
            Integer[] array = {2, 1};
            Integer[] expected = {1, 2};
            QuickSorter.sort(array);
            assertArrayEquals(expected, array);
        }

        @Test
        @DisplayName("两元素有序数组排序后不变")
        void shouldSortTwoElementOrderedArray() {
            Integer[] array = {1, 2};
            Integer[] expected = {1, 2};
            QuickSorter.sort(array);
            assertArrayEquals(expected, array);
        }

        @Test
        @DisplayName("字符串数组按字典序排序")
        void shouldSortStringArray() {
            String[] array = {"banana", "apple", "cherry", "date"};
            String[] expected = {"apple", "banana", "cherry", "date"};
            QuickSorter.sort(array);
            assertArrayEquals(expected, array);
        }

        @Test
        @DisplayName("超阈值大数组排序结果与 JDK 排序一致")
        void shouldSortLargeArrayConsistentWithJdk() {
            int size = 1000;
            Integer[] array = new Integer[size];
            for (int i = 0; i < size; i++) {
                array[i] = (size - i) % 97;
            }
            Integer[] expected = Arrays.copyOf(array, array.length);
            Arrays.sort(expected);

            QuickSorter.sort(array);
            assertArrayEquals(expected, array);
        }

        @Test
        @DisplayName("负数与零混合数组排序后升序")
        void shouldSortArrayWithNegativeAndZero() {
            Integer[] array = {0, -5, 3, -1, 2, -3, 5, -4, 1, -2, 4};
            Integer[] expected = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5};
            QuickSorter.sort(array);
            assertArrayEquals(expected, array);
        }
    }

    /**
     * 异常元素场景。
     */
    @Nested
    @DisplayName("异常元素")
    class NullElementTest {

        @Test
        @DisplayName("R02: 数组含 null 元素时抛出 NullPointerException")
        void shouldThrowNpeWhenArrayContainsNull() {
            Integer[] array = {3, null, 1};
            assertThrows(NullPointerException.class, () -> QuickSorter.sort(array));
        }
    }
}

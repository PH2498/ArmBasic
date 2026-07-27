package com.shuke.algo.sort;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * {@link QuickSort} 单元测试
 *
 * <p>遵循 FIRST 原则：快速、独立、可重复、自校验、及时。</p>
 *
 * @author shuke
 * @date 2026/07/27
 */
class QuickSortTest {

    /**
     * 普通乱序数组升序排序。
     */
    @Test
    void sort_unorderedArray_shouldBeAscending() {
        // given
        int[] input = {5, 3, 8, 1, 9, 2, 7, 4, 6};
        // when
        int[] result = QuickSort.sort(input);
        // then
        assertArrayEquals(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, result);
    }

    /**
     * 已升序数组排序后保持不变。
     */
    @Test
    void sort_alreadyAscending_shouldKeepOrder() {
        int[] input = {1, 2, 3, 4, 5};
        int[] result = QuickSort.sort(input);
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, result);
    }

    /**
     * 倒序数组排序后应为升序。
     */
    @Test
    void sort_descending_shouldBeAscending() {
        int[] input = {5, 4, 3, 2, 1};
        int[] result = QuickSort.sort(input);
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, result);
    }

    /**
     * 含重复元素数组排序后稳定排列。
     */
    @Test
    void sort_withDuplicates_shouldKeepAllElements() {
        int[] input = {3, 1, 2, 3, 1, 2};
        int[] result = QuickSort.sort(input);
        assertArrayEquals(new int[]{1, 1, 2, 2, 3, 3}, result);
    }

    /**
     * 单元素数组排序后保持不变。
     */
    @Test
    void sort_singleElement_shouldKeepUnchanged() {
        int[] input = {42};
        int[] result = QuickSort.sort(input);
        assertArrayEquals(new int[]{42}, result);
    }

    /**
     * 空数组排序后仍为空数组。
     */
    @Test
    void sort_emptyArray_shouldKeepEmpty() {
        int[] input = {};
        int[] result = QuickSort.sort(input);
        assertNotNull(result);
        assertArrayEquals(new int[]{}, result);
    }

    /**
     * 入参为 null 时应安全返回 null。
     */
    @Test
    void sort_nullInput_shouldReturnNull() {
        assertNull(QuickSort.sort(null));
    }

    /**
     * 原地排序应返回同一数组对象引用。
     */
    @Test
    void sort_sameReference_shouldBeInPlace() {
        int[] input = {2, 1, 3};
        int[] result = QuickSort.sort(input);
        assertSame(input, result);
    }

    /**
     * {@code sortedCopy} 不应修改原数组，并返回排序后的新数组。
     */
    @Test
    void sortedCopy_shouldNotMutateOriginal() {
        int[] input = {5, 3, 1, 2, 4};
        int[] copy = QuickSort.sortedCopy(input);
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, copy);
        assertArrayEquals(new int[]{5, 3, 1, 2, 4}, input);
    }

    /**
     * {@code sortedCopy} 对 null 入参应返回 null。
     */
    @Test
    void sortedCopy_nullInput_shouldReturnNull() {
        assertNull(QuickSort.sortedCopy(null));
    }

    /**
     * 较大随机数据排序结果应与 JDK 内置排序一致（交叉验证）。
     */
    @Test
    void sort_largeRandom_shouldMatchJdkSort() {
        int size = 1000;
        int[] input = new int[size];
        for (int i = 0; i < size; i++) {
            input[i] = (i * 73 + 37) % 997;
        }
        int[] expected = Arrays.copyOf(input, input.length);
        Arrays.sort(expected);
        int[] result = QuickSort.sort(input);
        assertArrayEquals(expected, result);
    }

    /**
     * 全部相同元素数组排序后保持不变。
     */
    @Test
    void sort_allSameElements_shouldKeepUnchanged() {
        int[] input = {7, 7, 7, 7, 7};
        int[] result = QuickSort.sort(input);
        assertArrayEquals(new int[]{7, 7, 7, 7, 7}, result);
    }

    /**
     * 负数与正数混合数组排序结果正确。
     */
    @Test
    void sort_negativeAndPositive_shouldBeAscending() {
        int[] input = {-3, 4, 0, -1, 2, -5};
        int[] result = QuickSort.sort(input);
        assertArrayEquals(new int[]{-5, -3, -1, 0, 2, 4}, result);
    }
}

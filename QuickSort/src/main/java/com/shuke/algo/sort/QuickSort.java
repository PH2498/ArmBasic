package com.shuke.algo.sort;

import java.util.Arrays;
import java.util.Objects;

/**
 * 快速排序算法工具类
 *
 * <p>采用 Lomuto 分区方案对整型数组进行原地排序，时间复杂度平均 O(n log n)，
 * 最坏 O(n^2)；空间复杂度 O(log n)（递归栈）。</p>
 *
 * @author shuke
 * @date 2026/07/27
 */
public class QuickSort {

    /**
     * 默认基准值选择下标偏移量，取区间末位元素作为基准。
     */
    private static final int PIVOT_TAIL_OFFSET = 0;

    /**
     * 最小可直接返回的子区间长度，长度小于等于该值时不再递归切分。
     */
    private static final int MIN_PARTITION_LENGTH = 1;

    /**
     * 私有构造方法，禁止实例化工具类。
     */
    private QuickSort() {
        // 工具类禁止实例化
    }

    /**
     * 对整型数组进行升序排序（原地修改）。
     *
     * @param array 待排序数组，允许为 null 或空数组
     * @return 排序后的数组引用，与入参为同一对象；入参为 null 时返回 null
     */
    public static int[] sort(int[] array) {
        if (Objects.isNull(array) || array.length <= MIN_PARTITION_LENGTH) {
            return array;
        }
        quickSort(array, 0, array.length - 1);
        return array;
    }

    /**
     * 对整型数组进行升序排序并返回新数组（不改原数组）。
     *
     * @param array 待排序数组，允许为 null 或空数组
     * @return 排序后的新数组；入参为 null 时返回 null
     */
    public static int[] sortedCopy(int[] array) {
        if (Objects.isNull(array)) {
            return null;
        }
        return sort(Arrays.copyOf(array, array.length));
    }

    /**
     * 递归对区间 [low, high] 进行快速排序。
     *
     * @param array 待排序数组
     * @param low   区间下界（含）
     * @param high  区间上界（含）
     */
    private static void quickSort(int[] array, int low, int high) {
        // 区间长度小于等于阈值时直接返回，避免无意义递归
        if (low >= high) {
            return;
        }
        int pivotIndex = partition(array, low, high);
        quickSort(array, low, pivotIndex - 1);
        quickSort(array, pivotIndex + 1, high);
    }

    /**
     * 以区间末位元素为基准进行 Lomuto 分区。
     *
     * <p>分区后基准左侧元素均不大于基准，右侧元素均大于基准。</p>
     *
     * @param array 待分区数组
     * @param low   区间下界（含）
     * @param high  区间上界（含），基准元素位于此下标
     * @return 分区后基准元素最终所在下标
     */
    private static int partition(int[] array, int low, int high) {
        int pivot = array[high + PIVOT_TAIL_OFFSET];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (array[j] <= pivot) {
                i++;
                swap(array, i, j);
            }
        }
        swap(array, i + 1, high);
        return i + 1;
    }

    /**
     * 交换数组中两个下标位置的元素。
     *
     * @param array 目标数组
     * @param i     下标 i
     * @param j     下标 j
     */
    private static void swap(int[] array, int i, int j) {
        if (i == j) {
            return;
        }
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

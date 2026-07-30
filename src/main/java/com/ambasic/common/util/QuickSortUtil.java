package com.ambasic.common.util;

import java.util.Arrays;
import java.util.Comparator;

/**
 * 快速排序工具类，提供对数组的原地快速排序实现。
 *
 * <p>该类为工具类，不可实例化，所有方法均为静态方法，直接通过类名访问。</p>
 *
 * <p>排序基于经典的快速排序算法（Lomuto 分区方案），平均时间复杂度 O(n log n)，
 * 最坏时间复杂度 O(n^2)，空间复杂度 O(log n)（递归栈）。</p>
 *
 * @author ArmBasic
 * @since 1.0.0
 */
public final class QuickSortUtil {

    /** 数组最小有效起始索引，用于参数校验。 */
    private static final int MIN_START_INDEX = 0;

    /** 默认比较器排序失败时的错误提示。 */
    private static final String ERROR_NULL_ARRAY = "待排序数组不能为 null";

    /** 比较器为空的错误提示。 */
    private static final String ERROR_NULL_COMPARATOR = "比较器不能为 null";

    /** 区间非法（起始大于结束）的错误提示。 */
    private static final String ERROR_INVALID_RANGE = "排序区间非法：startIndex 大于 endIndex";

    /**
     * 私有构造函数，防止工具类被实例化。
     */
    private QuickSortUtil() {
        throw new AssertionError("工具类禁止实例化");
    }

    /**
     * 对整型数组进行升序快速排序（原地修改）。
     *
     * @param array 待排序数组，不允许为 null
     * @throws IllegalArgumentException 当 array 为 null 时抛出
     */
    public static void sort(int[] array) {
        if (array == null) {
            throw new IllegalArgumentException(ERROR_NULL_ARRAY);
        }
        quickSortInt(array, MIN_START_INDEX, array.length - 1);
    }

    /**
     * 对整型数组的指定区间 [startIndex, endIndex] 进行升序快速排序（原地修改）。
     *
     * @param array      待排序数组，不允许为 null
     * @param startIndex 起始索引（包含），必须满足 0 <= startIndex < array.length
     * @param endIndex   结束索引（包含），必须满足 startIndex <= endIndex < array.length
     * @throws IllegalArgumentException 当 array 为 null 或区间非法时抛出
     */
    public static void sort(int[] array, int startIndex, int endIndex) {
        if (array == null) {
            throw new IllegalArgumentException(ERROR_NULL_ARRAY);
        }
        if (startIndex < MIN_START_INDEX || endIndex >= array.length || startIndex > endIndex) {
            throw new IllegalArgumentException(ERROR_INVALID_RANGE);
        }
        quickSortInt(array, startIndex, endIndex);
    }

    /**
     * 对泛型数组进行快速排序（原地修改），排序规则由 comparator 决定。
     *
     * @param <T>        数组元素类型
     * @param array      待排序数组，不允许为 null
     * @param comparator 比较器，不允许为 null
     * @throws IllegalArgumentException 当 array 或 comparator 为 null 时抛出
     */
    public static <T> void sort(T[] array, Comparator<? super T> comparator) {
        if (array == null) {
            throw new IllegalArgumentException(ERROR_NULL_ARRAY);
        }
        if (comparator == null) {
            throw new IllegalArgumentException(ERROR_NULL_COMPARATOR);
        }
        quickSortGeneric(array, MIN_START_INDEX, array.length - 1, comparator);
    }

    /**
     * 整型数组递归快速排序内部实现。
     *
     * @param array     待排序数组
     * @param low       当前分区起始索引
     * @param high      当前分区结束索引
     */
    private static void quickSortInt(int[] array, int low, int high) {
        if (low >= high) {
            return;
        }
        int pivotIndex = partitionInt(array, low, high);
        quickSortInt(array, low, pivotIndex - 1);
        quickSortInt(array, pivotIndex + 1, high);
    }

    /**
     * 整型数组分区操作，选取末位元素为基准值，返回分区后基准值最终位置。
     *
     * @param array 待分区数组
     * @param low   分区起始索引
     * @param high  分区结束索引
     * @return 基准值最终索引
     */
    private static int partitionInt(int[] array, int low, int high) {
        int pivot = array[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (array[j] <= pivot) {
                i++;
                swapInt(array, i, j);
            }
        }
        swapInt(array, i + 1, high);
        return i + 1;
    }

    /**
     * 交换整型数组中两个位置的元素。
     *
     * @param array 目标数组
     * @param i     第一个索引
     * @param j     第二个索引
     */
    private static void swapInt(int[] array, int i, int j) {
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    /**
     * 泛型数组递归快速排序内部实现。
     *
     * @param array      待排序数组
     * @param low        当前分区起始索引
     * @param high       当前分区结束索引
     * @param comparator 比较器
     */
    private static <T> void quickSortGeneric(T[] array, int low, int high, Comparator<? super T> comparator) {
        if (low >= high) {
            return;
        }
        int pivotIndex = partitionGeneric(array, low, high, comparator);
        quickSortGeneric(array, low, pivotIndex - 1, comparator);
        quickSortGeneric(array, pivotIndex + 1, high, comparator);
    }

    /**
     * 泛型数组分区操作，选取末位元素为基准值，返回分区后基准值最终位置。
     *
     * @param array      待分区数组
     * @param low        分区起始索引
     * @param high       分区结束索引
     * @param comparator 比较器
     * @return 基准值最终索引
     */
    private static <T> int partitionGeneric(T[] array, int low, int high, Comparator<? super T> comparator) {
        T pivot = array[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (comparator.compare(array[j], pivot) <= 0) {
                i++;
                swapGeneric(array, i, j);
            }
        }
        swapGeneric(array, i + 1, high);
        return i + 1;
    }

    /**
     * 交换泛型数组中两个位置的元素。
     *
     * @param array 目标数组
     * @param i     第一个索引
     * @param j     第二个索引
     */
    private static <T> void swapGeneric(T[] array, int i, int j) {
        T temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    /**
     * 创建数组副本并排序，不修改原数组。
     *
     * @param array 待排序数组，不允许为 null
     * @return 排序后的新数组
     * @throws IllegalArgumentException 当 array 为 null 时抛出
     */
    public static int[] sortedCopy(int[] array) {
        if (array == null) {
            throw new IllegalArgumentException(ERROR_NULL_ARRAY);
        }
        int[] copy = Arrays.copyOf(array, array.length);
        quickSortInt(copy, MIN_START_INDEX, copy.length - 1);
        return copy;
    }
}

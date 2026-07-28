package com.dtazziboot.algorithm.sort;

import java.util.Objects;

/**
 * 快速排序工具类。
 *
 * <p>基于分治思想实现的原地升序排序，平均时间复杂度 O(n log n)，空间复杂度 O(log n)（递归栈）。
 * 采用 Lomuto 分区方案，配合三数取中基准选择与小段插入排序退化，规避常见最坏退化场景。</p>
 *
 * <p>本类为无状态静态工具类，线程安全的前提是调用方保证传入数组在排序期间不被并发修改；
 * 算法本身无共享可变状态，并发安全由调用方自行保证。</p>
 *
 * @see java.lang.Comparable
 */
public final class QuickSorter {

    /**
     * 子数组长度阈值：长度不超过该值时切换为插入排序，以减少递归调用开销。
     */
    private static final int INSERTION_SORT_THRESHOLD = 16;

    /**
     * 工具类禁止实例化。
     */
    private QuickSorter() {
        throw new AssertionError("Utility class cannot be instantiated");
    }

    /**
     * 对泛型可比较数组进行原地升序排序。
     *
     * <p>排序后原数组即为升序，不产生新数组。对 null、空数组、单元素数组短路返回。
     * 若数组元素为 null，将在比较阶段抛出 {@link NullPointerException}。</p>
     *
     * @param array 待排序数组，要求元素非 null 且实现 {@link Comparable}
     * @param <T>   数组元素类型，须可比较
     * @throws NullPointerException 入参 array 为 null 时抛出（R01）
     */
    public static <T extends Comparable<? super T>> void sort(T[] array) {
        // R01：入参 array 不为 null
        Objects.requireNonNull(array, "array must not be null");

        // 短路：空数组或单元素数组无需排序
        int length = array.length;
        if (length <= 1) {
            return;
        }

        quickSortRecursive(array, 0, length - 1);
    }

    /**
     * 递归主流程：对数组指定区间 [low, high] 执行快速排序。
     *
     * <p>内部方法，不对外暴露边界校验，由 {@link #sort} 保证传入合法区间。</p>
     *
     * @param array 待排序数组
     * @param low   区间下界（含）
     * @param high  区间上界（含）
     * @param <T>   数组元素类型
     */
    private static <T> void quickSortRecursive(T[] array, int low, int high) {
        // R03：子数组长度 <= 1 时终止递归
        if (low >= high) {
            return;
        }

        // R04：子数组长度 <= 阈值时切换插入排序
        if (high - low + 1 <= INSERTION_SORT_THRESHOLD) {
            insertionSort(array, low, high);
            return;
        }

        // 选择基准并将其换至 high 端点
        choosePivot(array, low, high);

        // 分区，返回基准最终落点
        int pivotIndex = partition(array, low, high);

        // 递归处理基准左右两侧子区间
        quickSortRecursive(array, low, pivotIndex - 1);
        quickSortRecursive(array, pivotIndex + 1, high);
    }

    /**
     * 分区函数：基于 Lomuto 方案对区间 [low, high] 进行分区。
     *
     * <p>前置条件：基准元素已位于 high 端点（由 {@link #choosePivot} 完成）。
     * 分区不变量：[low, i] &lt; pivot，(i, j) &gt;= pivot，[j, high-1] 为未处理区段。
     * 分区后基准归位至最终落点并返回该索引。</p>
     *
     * @param array 待排序数组
     * @param low   区间下界（含）
     * @param high  区间上界（含），基准元素当前位置
     * @param <T>   数组元素类型
     * @return 基准最终落点索引
     */
    @SuppressWarnings("unchecked")
    private static <T> int partition(T[] array, int low, int high) {
        T pivot = array[high];
        // R02：若 pivot 为 null，此处 compareTo 将抛出 NullPointerException
        Comparable<? super T> comparablePivot = (Comparable<? super T>) pivot;
        int i = low - 1;

        for (int j = low; j < high; j++) {
            // R02：若 array[j] 为 null，此处 compareTo 将抛出 NullPointerException
            if (comparablePivot.compareTo(array[j]) > 0) {
                i++;
                swap(array, i, j);
            }
        }

        // 基准归位
        swap(array, i + 1, high);
        return i + 1;
    }

    /**
     * 基准选择：三数取中策略。
     *
     * <p>比较 low、mid、high 三处元素，取中值作为基准，并将基准交换至 high 端点，
     * 以配合 Lomuto 分区方案。返回值无实际意义，副作用为基准换至 high 端点。</p>
     *
     * @param array 待排序数组
     * @param low   区间下界（含）
     * @param high  区间上界（含）
     * @param <T>   数组元素类型
     */
    @SuppressWarnings("unchecked")
    private static <T> void choosePivot(T[] array, int low, int high) {
        int mid = low + (high - low) / 2;

        // 三数排序：经三次比较与交换，保证 array[low] <= array[mid] <= array[high]
        if (compareAt(array, low, mid) > 0) {
            swap(array, low, mid);
        }
        if (compareAt(array, low, high) > 0) {
            swap(array, low, high);
        }
        if (compareAt(array, mid, high) > 0) {
            swap(array, mid, high);
        }
        // 此时中值位于 mid，将其换至 high 端点作为 Lomuto 分区基准
        swap(array, mid, high);
    }

    /**
     * 元素交换：交换数组中 i 与 j 位置的元素。
     *
     * @param array 目标数组
     * @param i     索引 i
     * @param j     索引 j
     * @param <T>   数组元素类型
     */
    private static <T> void swap(T[] array, int i, int j) {
        T temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    /**
     * 对区间 [low, high] 执行插入排序，用于小段数据时规避递归开销。
     *
     * @param array 待排序数组
     * @param low   区间下界（含）
     * @param high  区间上界（含）
     * @param <T>   数组元素类型
     */
    @SuppressWarnings("unchecked")
    private static <T> void insertionSort(T[] array, int low, int high) {
        for (int i = low + 1; i <= high; i++) {
            T key = array[i];
            int j = i - 1;
            // 将 key 插入到 [low, i-1] 已排序段的正确位置
            while (j >= low && ((Comparable<? super T>) key).compareTo(array[j]) < 0) {
                array[j + 1] = array[j];
                j--;
            }
            array[j + 1] = key;
        }
    }

    /**
     * 比较数组中两个索引位置的元素，返回比较结果。
     *
     * @param array 目标数组
     * @param a     索引 a
     * @param b     索引 b
     * @param <T>   数组元素类型
     * @return array[a] 与 array[b] 的比较结果（负/零/正）
     */
    @SuppressWarnings("unchecked")
    private static <T> int compareAt(T[] array, int a, int b) {
        return ((Comparable<? super T>) array[a]).compareTo(array[b]);
    }
}

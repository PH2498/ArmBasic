package com.dtazziboot.algorithm.sort;

import java.util.Comparator;

/**
 * 堆排序算法库对外门面。
 *
 * <p>纯算法库，无外部中间件依赖。提供原地排序：
 * <ul>
 *     <li>自然序升序排序（{@link #sort(Comparable[])}，采用大顶堆）；</li>
 *     <li>自定义比较器排序（{@link #sort(Object[], Comparator)}，排序方向由比较器决定）。</li>
 * </ul>
 *
 * <p>时间复杂度：建堆 O(n)，整体排序 O(n log n)，最坏/平均/最好均为 O(n log n)。
 * 空间复杂度：原地排序，额外空间 O(1)（siftDown 采用迭代实现，无栈开销）。
 *
 * <p>线程安全：操作传入数组本身，不做共享状态封装；并发场景由调用方加锁。
 *
 * <p>稳定性：堆排序天然不稳定，本实现不做额外补偿。
 *
 * @author AI 编码引擎
 * @since 1.0.0
 */
public final class HeapSort {

    /** 工具类禁实例化。 */
    private HeapSort() {
    }

    /**
     * 按自然序升序排序（采用大顶堆，原地排序）。
     *
     * <p>等价于 {@code sort(arr, Comparator.naturalOrder())}。
     *
     * @param arr  待排序数组，原地修改
     * @param <T>  元素类型，需实现 {@link Comparable}
     * @throws IllegalArgumentException 当 {@code arr} 为 {@code null}
     */
    public static <T extends Comparable<? super T>> void sort(T[] arr) {
        if (arr == null) {
            throw new IllegalArgumentException("array must not be null");
        }
        if (arr.length < 2) {
            return;
        }
        doSort(arr, Comparator.naturalOrder());
    }

    /**
     * 按自定义比较器排序，原地修改数组。
     *
     * <p>排序方向完全由比较器决定：自然序比较器 → 升序；{@link Comparator#reverseOrder()} → 降序。
     *
     * @param arr        待排序数组，原地修改
     * @param comparator 比较器，决定排序方向与元素大小关系
     * @param <T>        元素类型
     * @throws IllegalArgumentException 当 {@code arr} 或 {@code comparator} 为 {@code null}
     */
    public static <T> void sort(T[] arr, Comparator<? super T> comparator) {
        if (arr == null) {
            throw new IllegalArgumentException("array must not be null");
        }
        if (comparator == null) {
            throw new IllegalArgumentException("comparator must not be null");
        }
        if (arr.length < 2) {
            return;
        }
        doSort(arr, comparator);
    }

    /**
     * 排序主流程：建堆 + 循环“取堆顶交换到末尾 → 缩堆 → siftDown”。
     *
     * @param arr 待排序数组
     * @param cmp 比较器
     * @param <T> 元素类型
     */
    private static <T> void doSort(T[] arr, Comparator<? super T> cmp) {
        int n = arr.length;
        buildHeap(arr, cmp);
        // end 为有效堆边界（exclusive），每次将堆顶极值交换到边界末端并缩堆
        for (int end = n - 1; end > 0; end--) {
            swap(arr, 0, end);
            siftDown(arr, 0, end, cmp);
        }
    }

    /**
     * 建堆：自最后一个非叶节点（n/2-1）向前批量下沉，复杂度 O(n)。
     *
     * @param arr 数组
     * @param cmp 比较器
     * @param <T> 元素类型
     */
    private static <T> void buildHeap(T[] arr, Comparator<? super T> cmp) {
        int n = arr.length;
        for (int i = n / 2 - 1; i >= 0; i--) {
            siftDown(arr, i, n, cmp);
        }
    }

    /**
     * 堆调整原语（下沉，迭代实现）。
     *
     * <p>规则：取左右子 {@code 2i+1}、{@code 2i+2}，在 {@code end}（exclusive）范围内选出较优子节点；
     * 若该子节点优于当前节点（{@code cmp.compare(child, parent) > 0}），交换并继续下沉，否则终止。
     * 只存在左子时仅与左子比较，不越界访问右子。比较相等（cmp == 0）时不交换。
     *
     * @param arr 数组
     * @param i   当前节点索引
     * @param end 有效堆边界（exclusive）
     * @param cmp 比较器
     * @param <T> 元素类型
     */
    private static <T> void siftDown(T[] arr, int i, int end, Comparator<? super T> cmp) {
        int parent = i;
        while (true) {
            int left = 2 * parent + 1;
            // 无子节点，终止
            if (left >= end) {
                break;
            }
            int largest = left;
            int right = left + 1;
            // 仅当右子在边界内且优于左子时选右子（H01-R05：不越界访问右子）
            if (right < end && cmp.compare(arr[right], arr[left]) > 0) {
                largest = right;
            }
            // 子节点优于当前父节点则下沉，否则终止（H05-R03：相等不交换，避免死循环）
            if (cmp.compare(arr[largest], arr[parent]) > 0) {
                swap(arr, parent, largest);
                parent = largest;
            } else {
                break;
            }
        }
    }

    /**
     * 交换数组中两个位置的元素。
     *
     * @param arr 数组
     * @param i   位置 i
     * @param j   位置 j
     * @param <T> 元素类型
     */
    private static <T> void swap(T[] arr, int i, int j) {
        T tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

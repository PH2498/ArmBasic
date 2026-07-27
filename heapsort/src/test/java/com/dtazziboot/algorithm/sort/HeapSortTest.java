package com.dtazziboot.algorithm.sort;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * {@link HeapSort} 单元测试，覆盖正常、边界、异常、比较器场景。
 *
 * <p>设计依据：F05 边界与参数校验、F06 用例覆盖。
 * <ul>
 *     <li>正常场景：随机数组、已排序数组、逆序数组、含重复元素数组、大数组。</li>
 *     <li>边界场景：null 数组、空数组、单元素数组。</li>
 *     <li>异常场景：comparator 为 null。</li>
 *     <li>比较器场景：降序排序、自定义对象排序。</li>
 * </ul>
 *
 * @author AI 编码引擎
 * @since 1.0.0
 */
public class HeapSortTest {

    /** 升序（自然序）排序：常规乱序整数数组。 */
    @Test
    public void testSortNaturalOrder() {
        Integer[] arr = {5, 2, 9, 1, 5, 6, 3, 8, 7, 0, 4};
        HeapSort.sort(arr);
        Integer[] expected = {0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9};
        assertArrayEquals(expected, arr);
    }

    /** 升序排序：已排序数组（最好情况，仍应保持有序）。 */
    @Test
    public void testSortAlreadySorted() {
        Integer[] arr = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        HeapSort.sort(arr);
        Integer[] expected = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        assertArrayEquals(expected, arr);
    }

    /** 升序排序：逆序数组（最坏情况之一）。 */
    @Test
    public void testSortReversed() {
        Integer[] arr = {9, 8, 7, 6, 5, 4, 3, 2, 1};
        HeapSort.sort(arr);
        Integer[] expected = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        assertArrayEquals(expected, arr);
    }

    /** 升序排序：含重复元素数组。 */
    @Test
    public void testSortWithDuplicates() {
        Integer[] arr = {3, 1, 2, 3, 1, 2, 3, 1, 2};
        HeapSort.sort(arr);
        Integer[] expected = {1, 1, 1, 2, 2, 2, 3, 3, 3};
        assertArrayEquals(expected, arr);
    }

    /** 升序排序：全部相同元素。 */
    @Test
    public void testSortAllEqual() {
        Integer[] arr = {7, 7, 7, 7, 7};
        HeapSort.sort(arr);
        Integer[] expected = {7, 7, 7, 7, 7};
        assertArrayEquals(expected, arr);
    }

    /** 边界：空数组应静默返回。 */
    @Test
    public void testSortEmptyArray() {
        Integer[] arr = {};
        HeapSort.sort(arr);
        assertEquals(0, arr.length);
    }

    /** 边界：单元素数组应静默返回。 */
    @Test
    public void testSortSingleElement() {
        Integer[] arr = {42};
        HeapSort.sort(arr);
        assertArrayEquals(new Integer[]{42}, arr);
    }

    /** 异常：null 数组抛 IllegalArgumentException。 */
    @Test
    public void testSortNullArrayThrows() {
        try {
            HeapSort.sort((Integer[]) null);
            fail("expected IllegalArgumentException for null array");
        } catch (IllegalArgumentException e) {
            assertEquals("array must not be null", e.getMessage());
        }
    }

    /** 异常：comparator 为 null 抛 IllegalArgumentException。 */
    @Test
    public void testSortNullComparatorThrows() {
        try {
            HeapSort.sort(new Integer[]{1, 2, 3}, null);
            fail("expected IllegalArgumentException for null comparator");
        } catch (IllegalArgumentException e) {
            assertEquals("comparator must not be null", e.getMessage());
        }
    }

    /** 异常：自然序重载下 null 数组抛 IllegalArgumentException。 */
    @Test
    public void testSortComparatorNullArrayThrows() {
        try {
            HeapSort.sort((Integer[]) null, Comparator.naturalOrder());
            fail("expected IllegalArgumentException for null array");
        } catch (IllegalArgumentException e) {
            assertEquals("array must not be null", e.getMessage());
        }
    }

    /** 比较器场景：降序排序（reverseOrder）。 */
    @Test
    public void testSortDescending() {
        Integer[] arr = {5, 2, 9, 1, 5, 6, 3, 8, 7, 0, 4};
        HeapSort.sort(arr, Comparator.reverseOrder());
        Integer[] expected = {9, 8, 7, 6, 5, 5, 4, 3, 2, 1, 0};
        assertArrayEquals(expected, arr);
    }

    /** 比较器场景：自定义对象按属性排序。 */
    @Test
    public void testSortCustomObjectsByField() {
        class Person {
            final String name;
            final int age;

            Person(String name, int age) {
                this.name = name;
                this.age = age;
            }
        }
        Person[] people = {
            new Person("alice", 30),
            new Person("bob", 25),
            new Person("carol", 35)
        };
        HeapSort.sort(people, Comparator.comparingInt(p -> p.age));
        assertEquals("bob", people[0].name);
        assertEquals("alice", people[1].name);
        assertEquals("carol", people[2].name);
    }

    /** 大数组场景：与系统库排序结果一致，验证正确性与复杂度可接受性。 */
    @Test
    public void testSortLargeArrayConsistency() {
        int size = 1000;
        Integer[] arr = new Integer[size];
        for (int i = 0; i < size; i++) {
            arr[i] = (i * 37 + 11) % size;
        }
        Integer[] expected = Arrays.copyOf(arr, arr.length);
        Arrays.sort(expected);
        HeapSort.sort(arr);
        assertArrayEquals(expected, arr);
    }

    /** 升序场景：负数与正数混合。 */
    @Test
    public void testSortNegativeAndPositive() {
        Integer[] arr = {3, -1, 0, -5, 2, -3, 5};
        HeapSort.sort(arr);
        Integer[] expected = {-5, -3, -1, 0, 2, 3, 5};
        assertArrayEquals(expected, arr);
    }

    /** 升序场景：String 自然序排序。 */
    @Test
    public void testSortStrings() {
        String[] arr = {"banana", "apple", "cherry", "date"};
        HeapSort.sort(arr);
        assertArrayEquals(new String[]{"apple", "banana", "cherry", "date"}, arr);
    }

    /** 原地修改验证：排序后数组引用不变（同一对象）。 */
    @Test
    public void testInPlaceModification() {
        Integer[] arr = {3, 1, 2};
        Integer[] ref = arr;
        HeapSort.sort(arr);
        assertArrayEquals(new Integer[]{1, 2, 3}, arr);
        assertEquals(ref, arr);
    }

    /** 比较器场景：空数组 + 自定义比较器，静默返回。 */
    @Test
    public void testSortEmptyWithComparator() {
        Integer[] arr = {};
        HeapSort.sort(arr, Collections.reverseOrder());
        assertEquals(0, arr.length);
    }

    /** 比较器场景：单元素 + 自定义比较器，静默返回。 */
    @Test
    public void testSortSingleWithComparator() {
        Integer[] arr = {42};
        HeapSort.sort(arr, Collections.reverseOrder());
        assertArrayEquals(new Integer[]{42}, arr);
    }
}

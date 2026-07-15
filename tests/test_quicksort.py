"""快速排序算法的单元测试。"""

import pytest

from src.sorting.quicksort import partition, quicksort, sort


class TestPartition:
    """partition 函数单元测试。"""

    def test_partition_basic(self):
        arr = [3, 1, 2]
        pi = partition(arr, 0, 2)
        assert pi == 1
        assert arr[pi] == 2
        assert arr[0] <= arr[pi]
        assert arr[2] > arr[pi]

    def test_partition_all_equal(self):
        arr = [5, 5, 5, 5]
        pi = partition(arr, 0, 3)
        assert pi == 3
        assert arr == [5, 5, 5, 5]

    def test_partition_single_element(self):
        arr = [7]
        pi = partition(arr, 0, 0)
        assert pi == 0
        assert arr == [7]


class TestQuicksort:
    """quicksort 函数单元测试。"""

    def test_empty_array(self):
        arr = []
        quicksort(arr, 0, -1)  # low > high，直接返回
        assert arr == []

    def test_single_element(self):
        arr = [1]
        quicksort(arr, 0, 0)
        assert arr == [1]

    def test_basic_sort(self):
        arr = [3, 1, 2]
        quicksort(arr, 0, 2)
        assert arr == [1, 2, 3]

    def test_all_equal(self):
        arr = [5, 5, 5, 5]
        quicksort(arr, 0, 3)
        assert arr == [5, 5, 5, 5]

    def test_already_sorted(self):
        arr = [1, 2, 3, 4, 5]
        quicksort(arr, 0, 4)
        assert arr == [1, 2, 3, 4, 5]

    def test_reverse_sorted(self):
        arr = [5, 4, 3, 2, 1]
        quicksort(arr, 0, 4)
        assert arr == [1, 2, 3, 4, 5]

    def test_negative_numbers(self):
        arr = [3, -1, 0, 2, -5]
        quicksort(arr, 0, 4)
        assert arr == [-5, -1, 0, 2, 3]

    def test_large_random(self):
        import random

        random.seed(42)
        arr = [random.randint(-100, 100) for _ in range(100)]
        expected = sorted(arr)
        quicksort(arr, 0, len(arr) - 1)
        assert arr == expected

    def test_type_error(self):
        with pytest.raises(TypeError):
            quicksort("not a list", 0, 0)

    def test_index_error_low(self):
        with pytest.raises(IndexError):
            quicksort([1, 2, 3], -1, 2)

    def test_index_error_high(self):
        with pytest.raises(IndexError):
            quicksort([1, 2, 3], 0, 3)


class TestSort:
    """sort 便捷函数单元测试。"""

    def test_sort_empty(self):
        assert sort([]) == []

    def test_sort_single(self):
        assert sort([42]) == [42]

    def test_sort_typical(self):
        assert sort([3, 1, 4, 1, 5, 9, 2, 6]) == [1, 1, 2, 3, 4, 5, 6, 9]

    def test_sort_returns_same_reference(self):
        arr = [3, 1, 2]
        result = sort(arr)
        assert result is arr

    def test_sort_type_error(self):
        with pytest.raises(TypeError):
            sort(123)
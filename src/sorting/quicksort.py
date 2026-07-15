"""快速排序（Quicksort）算法实现。

原地排序，以最右侧元素为 pivot，递归实现。
"""


def partition(arr: list[int], low: int, high: int) -> int:
    """以 arr[high] 为 pivot，将区间 [low, high] 分区。

    分区后，pivot 左侧元素均 ≤ pivot，右侧元素均 > pivot。
    返回 pivot 的最终索引位置。
    """
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def quicksort(arr: list[int], low: int, high: int) -> None:
    """对数组 arr 在闭区间 [low, high] 内进行原地快速排序。

    Args:
        arr: 待排序的整数列表。
        low: 区间左边界（含）。
        high: 区间右边界（含）。

    Raises:
        TypeError: 若 arr 不是 list 类型。
        IndexError: 若 low 或 high 越界。
    """
    if not isinstance(arr, list):
        raise TypeError(f"arr 必须是 list 类型，实际为 {type(arr).__name__}")

    if low < 0 or high >= len(arr):
        raise IndexError(
            f"索引越界：low={low}, high={high}, len(arr)={len(arr)}"
        )

    if low < high:
        pi = partition(arr, low, high)
        quicksort(arr, low, pi - 1)
        quicksort(arr, pi + 1, high)


def sort(arr: list[int]) -> list[int]:
    """对整数数组进行快速排序的便捷入口。

    返回排序后的数组引用（原地排序，同时返回以便链式调用）。
    """
    if not isinstance(arr, list):
        raise TypeError(f"arr 必须是 list 类型，实际为 {type(arr).__name__}")

    if len(arr) > 0:
        quicksort(arr, 0, len(arr) - 1)
    return arr
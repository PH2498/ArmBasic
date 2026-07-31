package com.antgroup.armbasic.demo.model;

import java.util.List;

/**
 * W03 冒泡排序出参 data。
 */
public class SortResult {

    private List<Integer> sorted;
    private int swapCount;

    public SortResult() {}

    public SortResult(List<Integer> sorted, int swapCount) {
        this.sorted = sorted;
        this.swapCount = swapCount;
    }

    public List<Integer> getSorted() { return sorted; }
    public void setSorted(List<Integer> sorted) { this.sorted = sorted; }

    public int getSwapCount() { return swapCount; }
    public void setSwapCount(int swapCount) { this.swapCount = swapCount; }
}

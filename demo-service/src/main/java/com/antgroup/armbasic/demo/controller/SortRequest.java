package com.antgroup.armbasic.demo.controller;

import java.util.List;

/**
 * W03 冒泡排序入参 DTO。
 */
public class SortRequest {
    private List<Integer> items;

    public List<Integer> getItems() { return items; }
    public void setItems(List<Integer> items) { this.items = items; }
}

package com.antgroup.armbasic.demo.controller;

/**
 * W04 导出入参 DTO。
 */
public class ExportRequest {
    private String tab;
    private String format;

    public String getTab() { return tab; }
    public void setTab(String tab) { this.tab = tab; }

    public String getFormat() { return format; }
    public void setFormat(String format) { this.format = format; }
}

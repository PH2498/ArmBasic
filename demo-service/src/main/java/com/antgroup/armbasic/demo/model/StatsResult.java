package com.antgroup.armbasic.demo.model;

/**
 * W05 调用统计出参 data。
 */
public class StatsResult {

    private String chartType;
    private String dimension;
    private java.util.List<SeriesItem> series;

    public StatsResult() {}

    public StatsResult(String chartType, String dimension, java.util.List<SeriesItem> series) {
        this.chartType = chartType;
        this.dimension = dimension;
        this.series = series;
    }

    public String getChartType() { return chartType; }
    public void setChartType(String chartType) { this.chartType = chartType; }

    public String getDimension() { return dimension; }
    public void setDimension(String dimension) { this.dimension = dimension; }

    public java.util.List<SeriesItem> getSeries() { return series; }
    public void setSeries(java.util.List<SeriesItem> series) { this.series = series; }

    /** 系列数据项 {name, value} */
    public static class SeriesItem {
        private String name;
        private long value;

        public SeriesItem() {}

        public SeriesItem(String name, long value) {
            this.name = name;
            this.value = value;
        }

        public String getName() { return name; }
        public void setName(String name) { this.name = name; }

        public long getValue() { return value; }
        public void setValue(long value) { this.value = value; }
    }
}

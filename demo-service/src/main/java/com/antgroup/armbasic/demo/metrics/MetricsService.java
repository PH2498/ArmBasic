package com.antgroup.armbasic.demo.metrics;

import com.antgroup.armbasic.demo.model.CallRecord;
import com.antgroup.armbasic.demo.model.DemoConstants;
import com.antgroup.armbasic.demo.model.DemoException;
import com.antgroup.armbasic.demo.model.StatsResult;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 统计聚合服务（内部接口 S04/S05）。
 * <p>
 * R06 dimension/chartType 枚举校验；line 按时间分桶（默认按天），pie/bar 按维度值聚合计数。
 */
@Service
public class MetricsService {

    @Autowired
    private MetricsStore metricsStore;

    /**
     * 记录埋点（S04）。
     */
    public void record(CallRecord record) {
        metricsStore.record(record);
    }

    /**
     * 聚合统计（S05）。
     *
     * @param dimension 维度 role/level/dept
     * @param chartType 图表类型 line/pie/bar
     * @return 聚合结果
     */
    public StatsResult aggregate(String dimension, String chartType) {
        // R06 枚举校验
        if (!DemoConstants.inEnum(dimension, DemoConstants.DIMENSIONS)) {
            throw new DemoException(DemoConstants.METRICS_001);
        }
        if (!DemoConstants.inEnum(chartType, DemoConstants.CHART_TYPES)) {
            throw new DemoException(DemoConstants.METRICS_002);
        }

        List<CallRecord> records = metricsStore.findAll();

        List<StatsResult.SeriesItem> series;
        if ("line".equals(chartType)) {
            // line 按时间分桶（默认按天）
            series = aggregateByDay(records);
        } else {
            // pie/bar 按维度值聚合计数
            series = aggregateByDimension(records, dimension);
        }

        return new StatsResult(chartType, dimension, series);
    }

    /**
     * 按维度值聚合计数（pie/bar）。
     */
    private List<StatsResult.SeriesItem> aggregateByDimension(List<CallRecord> records, String dimension) {
        Map<String, Long> counts = new LinkedHashMap<>();
        for (CallRecord r : records) {
            String key = extractDimensionValue(r, dimension);
            counts.merge(key, 1L, Long::sum);
        }
        return counts.entrySet().stream()
                .map(e -> new StatsResult.SeriesItem(e.getKey(), e.getValue()))
                .collect(Collectors.toList());
    }

    /**
     * 按天分桶聚合（line）。
     */
    private List<StatsResult.SeriesItem> aggregateByDay(List<CallRecord> records) {
        DateTimeFormatter fmt = DateTimeFormatter.ofPattern("yyyy-MM-dd");
        Map<String, Long> counts = new LinkedHashMap<>();
        for (CallRecord r : records) {
            String day = r.getCallTime() != null
                    ? r.getCallTime().toLocalDate().format(fmt)
                    : LocalDate.now(ZoneId.of(DemoConstants.ZONE_ID)).format(fmt);
            counts.merge(day, 1L, Long::sum);
        }
        return counts.entrySet().stream()
                .map(e -> new StatsResult.SeriesItem(e.getKey(), e.getValue()))
                .collect(Collectors.toList());
    }

    /**
     * 根据维度提取记录对应字段值。
     */
    private String extractDimensionValue(CallRecord r, String dimension) {
        switch (dimension) {
            case "role":
                return r.getRole();
            case "level":
                return r.getLevel();
            case "dept":
                return r.getDept();
            default:
                return "unknown";
        }
    }
}

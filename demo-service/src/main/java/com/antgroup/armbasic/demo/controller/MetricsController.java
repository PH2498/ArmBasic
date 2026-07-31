package com.antgroup.armbasic.demo.controller;

import com.antgroup.armbasic.demo.metrics.MetricsService;
import com.antgroup.armbasic.demo.model.ApiResult;
import com.antgroup.armbasic.demo.model.DemoConstants;
import com.antgroup.armbasic.demo.model.StatsResult;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * MetricsController — 承载 W05 调用统计接口。
 * <p>
 * 统计接口 /api/metrics/** 不埋点（避免自递归）。
 */
@RestController
@RequestMapping("/api/metrics")
public class MetricsController {

    @Autowired
    private MetricsService metricsService;

    /**
     * W05 调用统计。
     * GET /api/metrics/call-stats?dimension=&chartType=
     */
    @GetMapping("/call-stats")
    public ApiResult<StatsResult> callStats(
            @RequestParam String dimension,
            @RequestParam String chartType) {
        try {
            StatsResult result = metricsService.aggregate(dimension, chartType);
            return ApiResult.ok(result);
        } catch (IllegalArgumentException e) {
            String msg = e.getMessage();
            // 区分 METRICS_001 / METRICS_002
            if (DemoConstants.METRICS_001.equals(msg)) {
                return ApiResult.fail(DemoConstants.METRICS_001);
            } else if (DemoConstants.METRICS_002.equals(msg)) {
                return ApiResult.fail(DemoConstants.METRICS_002);
            }
            return ApiResult.fail(msg);
        } catch (Exception e) {
            return ApiResult.fail("METRICS_ERROR");
        }
    }
}

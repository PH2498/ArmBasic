package com.antgroup.armbasic.demo.controller;

import com.antgroup.armbasic.demo.metrics.MetricsService;
import com.antgroup.armbasic.demo.model.ApiResult;
import com.antgroup.armbasic.demo.model.DemoConstants;
import com.antgroup.armbasic.demo.model.DemoException;
import com.antgroup.armbasic.demo.model.StatsResult;
import lombok.extern.slf4j.Slf4j;
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
@Slf4j
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
        } catch (DemoException e) {
            // A6.2: 通过自定义异常的 errorCode 字段区分，不再使用字符串比较
            log.warn("callStats failed, illegal argument: dimension={}, chartType={}", dimension, chartType, e);
            return ApiResult.fail(e.getErrorCode());
        } catch (IllegalArgumentException e) {
            log.warn("callStats failed, illegal argument: dimension={}, chartType={}", dimension, chartType, e);
            return ApiResult.fail(e.getMessage());
        } catch (Exception e) {
            log.error("callStats failed, unexpected error: dimension={}, chartType={}", dimension, chartType, e);
            return ApiResult.fail(DemoConstants.DEMO_001);
        }
    }
}

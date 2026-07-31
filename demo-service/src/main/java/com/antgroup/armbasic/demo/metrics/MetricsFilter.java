package com.antgroup.armbasic.demo.metrics;

import com.antgroup.armbasic.demo.model.CallRecord;
import com.antgroup.armbasic.demo.model.DemoConstants;
import com.antgroup.armbasic.demo.model.PersonMeta;
import jakarta.servlet.Filter;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.ServletRequest;
import jakarta.servlet.ServletResponse;
import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.IOException;

/**
 * 埋点过滤器（MetricsFilter）。
 * <p>
 * 拦截 /api/demo/** 写入埋点记录。
 * 统计接口 /api/metrics/** 不埋点（避免自递归）。
 */
@Slf4j
@Component
public class MetricsFilter implements Filter {

    @Autowired
    private CallerResolver callerResolver;

    @Autowired
    private MetricsService metricsService;

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
            throws IOException, ServletException {
        chain.doFilter(request, response);

        if (request instanceof HttpServletRequest httpRequest) {
            String path = httpRequest.getRequestURI();
            // 埋点范围：所有 /api/demo/** 与 /api/demo/export，不含 /api/metrics/**
            if (path != null && path.startsWith(DemoConstants.METRICS_PATH_PREFIX)) {
                recordCall(httpRequest);
            }
        }
    }

    /**
     * 记录调用埋点。
     */
    private void recordCall(HttpServletRequest request) {
        try {
            PersonMeta person = callerResolver.resolve(request);
            CallRecord record = new CallRecord();
            record.setApiName(request.getMethod() + " " + request.getRequestURI());
            record.setCallerId(person.getCallerId());
            record.setCallerName(person.getCallerName());
            record.setRole(person.getRole());
            record.setLevel(person.getLevel());
            record.setDept(person.getDept());
            metricsService.record(record);
        } catch (Exception e) {
            // 埋点失败不影响主流程
            // 不记录请求原文（hash raw 可能为敏感输入），仅记录异常
            log.warn("metrics record failed", e);
        }
    }
}

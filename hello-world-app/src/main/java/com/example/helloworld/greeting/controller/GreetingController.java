package com.example.helloworld.greeting.controller;

import com.example.helloworld.common.api.Result;
import com.example.helloworld.greeting.config.GreetingProperties;
import com.example.helloworld.greeting.service.GreetingService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 问候接口控制器。
 *
 * <p>对外提供 GET /api/hello 接口，返回问候语。</p>
 */
@RestController
@RequestMapping("/api")
public class GreetingController {

    private static final String MAINTENANCE_MESSAGE = "服务维护中，请稍后再试";

    private final GreetingService greetingService;
    private final GreetingProperties greetingProperties;

    public GreetingController(GreetingService greetingService, GreetingProperties greetingProperties) {
        this.greetingService = greetingService;
        this.greetingProperties = greetingProperties;
    }

    /**
     * 查询问候语。
     *
     * <p>当应急开关 {@code greeting.enabled=false} 时返回维护降级提示。</p>
     *
     * @return 统一响应体，data 为问候语文本
     */
    @GetMapping("/hello")
    public Result<String> hello() {
        if (!greetingProperties.enabled()) {
            return Result.maintenance(MAINTENANCE_MESSAGE);
        }
        return Result.success(greetingService.getGreeting());
    }
}

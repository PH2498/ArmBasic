package com.example.helloworld.greeting.service.impl;

import com.example.helloworld.greeting.config.GreetingProperties;
import com.example.helloworld.greeting.service.GreetingService;
import org.springframework.stereotype.Service;

/**
 * 问候服务实现。
 *
 * <p>问候语来源于配置项 {@code greeting.message}，默认为 {@code "Hello, World!"}。</p>
 */
@Service
public class GreetingServiceImpl implements GreetingService {

    /** 默认问候语，配置缺失时兜底 */
    private static final String DEFAULT_GREETING = "Hello, World!";

    private final GreetingProperties greetingProperties;

    public GreetingServiceImpl(GreetingProperties greetingProperties) {
        this.greetingProperties = greetingProperties;
    }

    @Override
    public String getGreeting() {
        String message = greetingProperties.message();
        if (message == null || message.isBlank()) {
            return DEFAULT_GREETING;
        }
        return message;
    }
}

package com.example.helloworld.greeting.service.impl;

import com.example.helloworld.greeting.config.GreetingProperties;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * {@link GreetingServiceImpl} 单元测试。
 */
class GreetingServiceImplTest {

    @Test
    @DisplayName("配置正常时返回配置的问候语")
    void shouldReturnConfiguredMessage() {
        GreetingProperties properties = new GreetingProperties(true, "Hello, World!");
        GreetingServiceImpl service = new GreetingServiceImpl(properties);

        assertEquals("Hello, World!", service.getGreeting());
    }

    @Test
    @DisplayName("问候语为空时兜底返回默认问候语")
    void shouldReturnDefaultWhenMessageBlank() {
        GreetingProperties properties = new GreetingProperties(true, "");
        GreetingServiceImpl service = new GreetingServiceImpl(properties);

        assertEquals("Hello, World!", service.getGreeting());
    }

    @Test
    @DisplayName("问候语为 null 时兜底返回默认问候语")
    void shouldReturnDefaultWhenMessageNull() {
        GreetingProperties properties = new GreetingProperties(true, null);
        GreetingServiceImpl service = new GreetingServiceImpl(properties);

        assertEquals("Hello, World!", service.getGreeting());
    }

    @Test
    @DisplayName("问候语为纯空白时兜底返回默认问候语")
    void shouldReturnDefaultWhenMessageWhitespace() {
        GreetingProperties properties = new GreetingProperties(true, "   ");
        GreetingServiceImpl service = new GreetingServiceImpl(properties);

        assertEquals("Hello, World!", service.getGreeting());
    }
}

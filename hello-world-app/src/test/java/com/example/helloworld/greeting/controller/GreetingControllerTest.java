package com.example.helloworld.greeting.controller;

import com.example.helloworld.common.api.Result;
import com.example.helloworld.greeting.config.GreetingProperties;
import com.example.helloworld.greeting.service.GreetingService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * {@link GreetingController} 单元测试。
 */
class GreetingControllerTest {

    @Test
    @DisplayName("应急开关开启时返回问候语")
    void shouldReturnGreetingWhenEnabled() {
        GreetingService greetingService = mock(GreetingService.class);
        when(greetingService.getGreeting()).thenReturn("Hello, World!");
        GreetingProperties properties = new GreetingProperties(true, "Hello, World!");
        GreetingController controller = new GreetingController(greetingService, properties);

        Result<String> result = controller.hello();

        assertEquals(0, result.code());
        assertEquals("Hello, World!", result.data());
    }

    @Test
    @DisplayName("应急开关关闭时返回维护降级响应")
    void shouldReturnMaintenanceWhenDisabled() {
        GreetingService greetingService = mock(GreetingService.class);
        GreetingProperties properties = new GreetingProperties(false, "Hello, World!");
        GreetingController controller = new GreetingController(greetingService, properties);

        Result<String> result = controller.hello();

        assertEquals(503, result.code());
        assertEquals("服务维护中，请稍后再试", result.message());
        assertNull(result.data());
    }
}

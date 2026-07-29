package com.example.helloworld.greeting.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Greeting 问候模块配置属性。
 *
 * @param enabled  应急开关，关闭时接口返回维护提示
 * @param message  问候语内容
 */
@ConfigurationProperties(prefix = "greeting")
public record GreetingProperties(boolean enabled, String message) {
}

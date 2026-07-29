package com.example.helloworld.greeting.config;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Configuration;

/**
 * Greeting 模块配置类。
 *
 * <p>启用 {@link GreetingProperties} 的属性绑定。</p>
 */
@Configuration
@EnableConfigurationProperties(GreetingProperties.class)
public class GreetingConfiguration {
}

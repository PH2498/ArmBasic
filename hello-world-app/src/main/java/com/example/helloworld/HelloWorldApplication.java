package com.example.helloworld;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Hello World 问候服务启动类。
 *
 * <p>作为最小可运行示例的工程入口，负责引导 Spring 容器与内嵌 Web 容器启动。</p>
 */
@SpringBootApplication
public class HelloWorldApplication {

    public static void main(String[] args) {
        SpringApplication.run(HelloWorldApplication.class, args);
    }
}

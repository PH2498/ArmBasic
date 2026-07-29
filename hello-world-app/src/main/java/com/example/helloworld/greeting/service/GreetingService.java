package com.example.helloworld.greeting.service;

/**
 * 问候服务。
 *
 * <p>负责构造并返回问候语，为核心业务能力，无外部依赖。</p>
 */
public interface GreetingService {

    /**
     * 获取问候语。
     *
     * @return 问候语文本
     */
    String getGreeting();
}

package com.armbasic.helloworld;

/**
 * HelloWorld 程序入口类。
 * <p>
 * 用于演示符合编码规范的最小可运行 Java 程序，启动后向标准输出打印问候语。
 * </p>
 *
 * @author ArmBasic
 * @version 1.0.0
 */
public class HelloWorld {

    /**
     * 默认问候语常量。
     */
    private static final String DEFAULT_GREETING = "hello world";

    /**
     * 程序入口方法，启动后输出默认问候语。
     *
     * @param args 启动参数，当前实现未使用
     */
    public static void main(String[] args) {
        System.out.println(DEFAULT_GREETING);
    }
}

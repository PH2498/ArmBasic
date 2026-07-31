package com.antgroup.armbasic.demo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * 算法演示与调用埋点可视化 - 后端服务入口。
 * <p>
 * 承载五个接口：
 * <ul>
 *   <li>W01 GET  /api/demo/hello  — HelloWorld</li>
 *   <li>W02 POST /api/demo/hash   — 哈希算法</li>
 *   <li>W03 POST /api/demo/sort   — 冒泡排序</li>
 *   <li>W04 POST /api/demo/export — 导出</li>
 *   <li>W05 GET  /api/metrics/call-stats — 调用统计</li>
 * </ul>
 */
@SpringBootApplication
public class DemoServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(DemoServiceApplication.class, args);
    }
}

package com.example.helloworld.common.api;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * {@link Result} 单元测试。
 */
class ResultTest {

    @Test
    @DisplayName("success 成功响应码为 0")
    void shouldBuildSuccessResult() {
        Result<String> result = Result.success("Hello, World!");

        assertEquals(0, result.code());
        assertEquals("success", result.message());
        assertEquals("Hello, World!", result.data());
    }

    @Test
    @DisplayName("maintenance 维护响应码为 503 且 data 为 null")
    void shouldBuildMaintenanceResult() {
        Result<String> result = Result.maintenance("维护中");

        assertEquals(503, result.code());
        assertEquals("维护中", result.message());
        assertNull(result.data());
    }
}

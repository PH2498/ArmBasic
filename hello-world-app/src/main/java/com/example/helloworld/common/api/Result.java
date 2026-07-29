package com.example.helloworld.common.api;

import java.io.Serializable;

/**
 * 统一响应体。
 *
 * <p>对外/内部接口统一返回 {@code {code, message, data}} 结构，便于调用方按统一契约解析。</p>
 *
 * @param code    业务状态码，0 表示成功
 * @param message 提示信息
 * @param data    业务数据
 * @param <T>     业务数据类型
 */
public record Result<T>(int code, String message, T data) implements Serializable {

    /** 成功状态码 */
    public static final int SUCCESS_CODE = 0;

    /** 维护/降级状态码 */
    public static final int MAINTENANCE_CODE = 503;

    private Result(int code, String message, T data) {
        this.code = code;
        this.message = message;
        this.data = data;
    }

    /**
     * 构造成功响应。
     *
     * @param data 业务数据
     * @param <T>  业务数据类型
     * @return 成功响应
     */
    public static <T> Result<T> success(T data) {
        return new Result<>(SUCCESS_CODE, "success", data);
    }

    /**
     * 构造成功响应（自定义提示信息）。
     *
     * @param message 提示信息
     * @param data    业务数据
     * @param <T>     业务数据类型
     * @return 成功响应
     */
    public static <T> Result<T> success(String message, T data) {
        return new Result<>(SUCCESS_CODE, message, data);
    }

    /**
     * 构造维护降级响应。
     *
     * @param message 维护提示信息
     * @param <T>     业务数据类型
     * @return 维护降级响应
     */
    public static <T> Result<T> maintenance(String message) {
        return new Result<>(MAINTENANCE_CODE, message, null);
    }
}

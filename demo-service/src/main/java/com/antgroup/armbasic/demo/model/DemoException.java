package com.antgroup.armbasic.demo.model;

/**
 * 业务异常，携带错误码。
 * <p>
 * 用于替代通过 e.getMessage() 字符串比较区分错误码的模式（A6.2 改进）。
 * 调用方可通过 {@link #getErrorCode()} 直接获取错误码，无需字符串匹配。
 */
public class DemoException extends RuntimeException {

    private final String errorCode;

    public DemoException(String errorCode) {
        super(errorCode);
        this.errorCode = errorCode;
    }

    public String getErrorCode() {
        return errorCode;
    }
}

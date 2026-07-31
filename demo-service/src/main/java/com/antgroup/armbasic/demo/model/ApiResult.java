package com.antgroup.armbasic.demo.model;

/**
 * 通用出参结构：{ "result": "OK"|"FAIL", "msg": "SUCCESS"|错误描述, "data": {} }
 */
public class ApiResult<T> {

    private String result;
    private String msg;
    private T data;

    private ApiResult(String result, String msg, T data) {
        this.result = result;
        this.msg = msg;
        this.data = data;
    }

    public static <T> ApiResult<T> ok(T data) {
        return new ApiResult<>("OK", "SUCCESS", data);
    }

    public static <T> ApiResult<T> fail(String msg) {
        return new ApiResult<>("FAIL", msg, null);
    }

    public String getResult() {
        return result;
    }

    public void setResult(String result) {
        this.result = result;
    }

    public String getMsg() {
        return msg;
    }

    public void setMsg(String msg) {
        this.msg = msg;
    }

    public T getData() {
        return data;
    }

    public void setData(T data) {
        this.data = data;
    }
}

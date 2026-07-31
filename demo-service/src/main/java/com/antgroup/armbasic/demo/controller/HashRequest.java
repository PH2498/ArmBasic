package com.antgroup.armbasic.demo.controller;

/**
 * W02 哈希算法入参 DTO。
 */
public class HashRequest {
    private String raw;
    private String algorithm;

    public String getRaw() { return raw; }
    public void setRaw(String raw) { this.raw = raw; }

    public String getAlgorithm() { return algorithm; }
    public void setAlgorithm(String algorithm) { this.algorithm = algorithm; }
}

package com.antgroup.armbasic.demo.model;

/**
 * 人员维度元数据（mock）。
 * <p>
 * 字段：callerId / callerName / role（人员类型） / level（人员层级） / dept（人员部门）。
 */
public class PersonMeta {

    private String callerId;
    private String callerName;
    private String role;
    private String level;
    private String dept;

    public PersonMeta() {}

    public PersonMeta(String callerId, String callerName, String role, String level, String dept) {
        this.callerId = callerId;
        this.callerName = callerName;
        this.role = role;
        this.level = level;
        this.dept = dept;
    }

    public String getCallerId() { return callerId; }
    public void setCallerId(String callerId) { this.callerId = callerId; }

    public String getCallerName() { return callerName; }
    public void setCallerName(String callerName) { this.callerName = callerName; }

    public String getRole() { return role; }
    public void setRole(String role) { this.role = role; }

    public String getLevel() { return level; }
    public void setLevel(String level) { this.level = level; }

    public String getDept() { return dept; }
    public void setDept(String dept) { this.dept = dept; }
}

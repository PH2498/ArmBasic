package com.antgroup.armbasic.demo.model;

import java.time.LocalDateTime;
import java.time.ZoneId;

/**
 * 接口调用埋点记录。
 * <p>
 * 对应表结构 call_record（MVP 内存存储，可选 H2 落盘）。
 */
public class CallRecord {

    private Long id;
    private LocalDateTime callTime;
    private String apiName;
    private String callerId;
    private String callerName;
    private String role;
    private String level;
    private String dept;
    private LocalDateTime gmtCreate;

    public CallRecord() {
        this.gmtCreate = LocalDateTime.now(ZoneId.of(DemoConstants.ZONE_ID));
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public LocalDateTime getCallTime() { return callTime; }
    public void setCallTime(LocalDateTime callTime) { this.callTime = callTime; }

    public String getApiName() { return apiName; }
    public void setApiName(String apiName) { this.apiName = apiName; }

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

    public LocalDateTime getGmtCreate() { return gmtCreate; }
    public void setGmtCreate(LocalDateTime gmtCreate) { this.gmtCreate = gmtCreate; }
}

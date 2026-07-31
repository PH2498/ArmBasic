package com.antgroup.armbasic.demo.metrics;

import com.antgroup.armbasic.demo.model.DemoConstants;
import com.antgroup.armbasic.demo.model.PersonMeta;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * 调用人解析器（内部接口 S06）。
 * <p>
 * MVP 从请求头 X-Caller-Id 读取，缺失则用 mock 池轮询分配（A08 假设）。
 */
@Component
public class CallerResolver {

    @Autowired
    private PersonMetaRepository personMetaRepository;

    /**
     * 解析调用人元数据。
     *
     * @param request HTTP 请求
     * @return 人员维度元数据
     */
    public PersonMeta resolve(HttpServletRequest request) {
        String callerId = request.getHeader(DemoConstants.CALLER_HEADER);
        return personMetaRepository.findById(callerId);
    }
}

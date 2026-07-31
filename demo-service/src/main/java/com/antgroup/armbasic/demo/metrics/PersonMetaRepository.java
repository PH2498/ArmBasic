package com.antgroup.armbasic.demo.metrics;

import com.antgroup.armbasic.demo.model.PersonMeta;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

/**
 * 人员元数据查询（集成层 I01）。
 * <p>
 * MVP 内存 mock 池，后续可替换为组织主数据 RPC。
 */
@Component
public class PersonMetaRepository {

    private final Map<String, PersonMeta> mockPool = new HashMap<>();

    public PersonMetaRepository() {
        // mock 人员元数据池（callerId → 人员维度元数据）
        mockPool.put("u001", new PersonMeta("u001", "张三", "研发", "P7", "研发部"));
        mockPool.put("u002", new PersonMeta("u002", "李四", "产品", "P6", "产品部"));
        mockPool.put("u003", new PersonMeta("u003", "王五", "测试", "P5", "测试部"));
        mockPool.put("u004", new PersonMeta("u004", "赵六", "研发", "P8", "研发部"));
        mockPool.put("u005", new PersonMeta("u005", "钱七", "产品", "P7", "产品部"));
        mockPool.put("u006", new PersonMeta("u006", "孙八", "研发", "P6", "研发部"));
        mockPool.put("u007", new PersonMeta("u007", "周九", "测试", "P9", "测试部"));
        mockPool.put("u008", new PersonMeta("u008", "吴十", "研发", "P5", "研发部"));
    }

    /**
     * 根据 callerId 查询人员元数据。
     *
     * @param callerId 调用人 ID
     * @return 人员元数据，缺失时返回 mock 默认值
     */
    public PersonMeta findById(String callerId) {
        if (callerId != null && mockPool.containsKey(callerId)) {
            return mockPool.get(callerId);
        }
        // 缺失时轮询分配一个 mock 人员（A08 假设）
        // 使用位掩码避免 Math.abs(Integer.MIN_VALUE) 返回负数导致数组越界（B029 修复）
        int idx = callerId == null ? 0 : (callerId.hashCode() & 0x7fffffff) % mockPool.size();
        String key = (String) mockPool.keySet().toArray()[idx];
        return mockPool.get(key);
    }
}

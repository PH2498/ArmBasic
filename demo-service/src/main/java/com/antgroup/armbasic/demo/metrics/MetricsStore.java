package com.antgroup.armbasic.demo.metrics;

import com.antgroup.armbasic.demo.model.CallRecord;
import com.antgroup.armbasic.demo.model.DemoConstants;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 埋点存储（MVP 内存 ConcurrentLinkedQueue）。
 * <p>
 * A04 假设：MVP 内存存储，可选 H2/SQLite 落盘。
 */
@Component
public class MetricsStore {

    private final ConcurrentLinkedQueue<CallRecord> store = new ConcurrentLinkedQueue<>();
    private final AtomicLong idSeq = new AtomicLong(0);

    /**
     * 记录一条埋点（S04）。
     */
    public void record(CallRecord record) {
        record.setId(idSeq.incrementAndGet());
        if (record.getCallTime() == null) {
            record.setCallTime(LocalDateTime.now(ZoneId.of(DemoConstants.ZONE_ID)));
        }
        store.add(record);
    }

    /**
     * 获取全部埋点记录（用于聚合统计）。
     */
    public List<CallRecord> findAll() {
        return new ArrayList<>(store);
    }
}

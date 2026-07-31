package com.antgroup.armbasic.demo.model;

/**
 * 枚举与常量定义（对应 design.md 5.1.1.x）。
 */
public final class DemoConstants {

    private DemoConstants() {}

    /** 哈希算法枚举 */
    public static final String[] HASH_ALGORITHMS = {"md5", "sha1", "sha256"};
    public static final String DEFAULT_HASH_ALGORITHM = "sha256";

    /** 导出格式枚举 */
    public static final String[] EXPORT_FORMATS = {"csv", "xlsx"};
    public static final String DEFAULT_EXPORT_FORMAT = "csv";

    /** 导出 Tab 枚举 */
    public static final String[] EXPORT_TABS = {"hello", "hash", "sort"};

    /** 图表类型枚举 */
    public static final String[] CHART_TYPES = {"line", "pie", "bar"};

    /** 统计维度枚举 */
    public static final String[] DIMENSIONS = {"role", "level", "dept"};

    /** 埋点覆盖路径前缀 */
    public static final String METRICS_PATH_PREFIX = "/api/demo";

    /** 调用人请求头 */
    public static final String CALLER_HEADER = "X-Caller-Id";

    /** HelloWorld 固定文案 */
    public static final String HELLO_WORLD_MESSAGE = "HelloWorld";

    /** XLSX Content-Type */
    public static final String XLSX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

    /** CSV Content-Type */
    public static final String CSV_CONTENT_TYPE = "text/csv";

    /** 导出样例数据（A8.3: 消除魔法值，统一管理导出样例输入） */
    public static final String EXPORT_SAMPLE_RAW = "hello";
    public static final String EXPORT_SAMPLE_ALGORITHM = "sha256";
    public static final int[] EXPORT_SAMPLE_SORT_ITEMS = {5, 3, 8, 1, 2};

    /** 统一时区 */
    public static final String ZONE_ID = "Asia/Shanghai";

    // ---- 错误码 ----
    public static final String DEMO_001 = "DEMO_001"; // 服务内部异常
    public static final String DEMO_002 = "DEMO_002"; // raw 为空
    public static final String DEMO_003 = "DEMO_003"; // algorithm 不支持
    public static final String DEMO_004 = "DEMO_004"; // items 为空或非数组
    public static final String DEMO_005 = "DEMO_005"; // tab 不支持
    public static final String DEMO_006 = "DEMO_006"; // format 不支持
    public static final String METRICS_001 = "METRICS_001"; // dimension 不支持
    public static final String METRICS_002 = "METRICS_002"; // chartType 不支持

    /**
     * 校验值是否在枚举集合中。
     */
    public static boolean inEnum(String value, String[] enumSet) {
        if (value == null) {
            return false;
        }
        for (String e : enumSet) {
            if (e.equalsIgnoreCase(value)) {
                return true;
            }
        }
        return false;
    }
}

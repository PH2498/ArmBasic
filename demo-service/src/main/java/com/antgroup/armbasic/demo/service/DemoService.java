package com.antgroup.armbasic.demo.service;

import com.antgroup.armbasic.demo.model.DemoConstants;
import com.antgroup.armbasic.demo.model.SortResult;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;

/**
 * DemoService 模块业务逻辑（内部接口 S01/S02）。
 * <p>
 * S01: String hash(String raw, String algorithm)
 * S02: SortResult bubbleSort(List<Integer> items)
 */
@Slf4j
@Service
public class DemoService {

    /**
     * 哈希计算（S01）。
     * <p>
     * R01 raw 非空校验；R02 algorithm 不在枚举集时回退 sha256。
     *
     * @param raw       原文
     * @param algorithm 算法 md5/sha1/sha256，默认 sha256
     * @return [实际算法, 十六进制摘要]
     */
    public String[] hash(String raw, String algorithm) {
        // R01 raw 非空校验
        if (raw == null || raw.isEmpty()) {
            throw new IllegalArgumentException(DemoConstants.DEMO_002);
        }

        // R02 algorithm 不在枚举集时回退 sha256
        String actualAlgorithm = DemoConstants.DEFAULT_HASH_ALGORITHM;
        if (algorithm != null && DemoConstants.inEnum(algorithm, DemoConstants.HASH_ALGORITHMS)) {
            actualAlgorithm = algorithm.toLowerCase();
        }

        try {
            MessageDigest md = MessageDigest.getInstance(actualAlgorithm.toUpperCase());
            byte[] digest = md.digest(raw.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            String hex = HexFormat.of().formatHex(digest);
            return new String[]{actualAlgorithm, hex};
        } catch (NoSuchAlgorithmException e) {
            log.error("hash failed, unsupported algorithm: {}", actualAlgorithm, e);
            throw new RuntimeException(DemoConstants.DEMO_001, e);
        }
    }

    /**
     * 冒泡排序（S02）。
     * <p>
     * R03 items 非空校验；稳定升序；统计实际交换次数。
     *
     * @param items 待排序整数数组
     * @return 排序结果 + 交换次数
     */
    public SortResult bubbleSort(List<Integer> items) {
        // R03 items 非空校验
        if (items == null || items.isEmpty()) {
            throw new IllegalArgumentException(DemoConstants.DEMO_004);
        }

        List<Integer> arr = new ArrayList<>(items);
        int swapCount = 0;
        int n = arr.size();

        for (int i = 0; i < n - 1; i++) {
            boolean swapped = false;
            for (int j = 0; j < n - 1 - i; j++) {
                if (arr.get(j) > arr.get(j + 1)) {
                    // 交换
                    int temp = arr.get(j);
                    arr.set(j, arr.get(j + 1));
                    arr.set(j + 1, temp);
                    swapCount++;
                    swapped = true;
                }
            }
            // 优化：若本轮无交换则已有序
            if (!swapped) {
                break;
            }
        }

        return new SortResult(arr, swapCount);
    }
}

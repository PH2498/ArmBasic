package com.antgroup.armbasic.demo.controller;

import com.antgroup.armbasic.demo.model.ApiResult;
import com.antgroup.armbasic.demo.model.DemoConstants;
import com.antgroup.armbasic.demo.model.SortResult;
import com.antgroup.armbasic.demo.service.DemoService;
import com.antgroup.armbasic.demo.service.ExportService;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.OutputStream;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * DemoController — 承载 W01~W04 四个接口。
 * <p>
 * 代码注释中的 R 编号（R01~R06）来源于系统设计文档 .agents/system.changes/design.md 的校验规则章节，
 * 其中 R01~R03 为 demo 接口校验规则，R04~R05 为导出校验规则，R06 为统计校验规则。
 */
@Slf4j
@RestController
@RequestMapping("/api/demo")
public class DemoController {

    @Autowired
    private DemoService demoService;

    @Autowired
    private ExportService exportService;

    /**
     * W01 HelloWorld。
     * GET /api/demo/hello
     */
    @GetMapping("/hello")
    public ApiResult<Map<String, String>> hello() {
        Map<String, String> data = new HashMap<>();
        data.put("message", DemoConstants.HELLO_WORLD_MESSAGE);
        return ApiResult.ok(data);
    }

    /**
     * W02 哈希算法。
     * POST /api/demo/hash
     */
    @PostMapping("/hash")
    public ApiResult<Map<String, String>> hash(@RequestBody HashRequest request) {
        try {
            String raw = request.getRaw();
            String algorithm = request.getAlgorithm();
            // R01 校验在 Service 层
            String[] result = demoService.hash(raw, algorithm);
            Map<String, String> data = new HashMap<>();
            data.put("algorithm", result[0]);
            data.put("digest", result[1]);
            return ApiResult.ok(data);
        } catch (IllegalArgumentException e) {
            log.warn("hash failed, illegal argument", e);
            return ApiResult.fail(e.getMessage());
        } catch (Exception e) {
            log.error("hash failed, unexpected error", e);
            return ApiResult.fail(DemoConstants.DEMO_001);
        }
    }

    /**
     * W03 冒泡排序。
     * POST /api/demo/sort
     */
    @PostMapping("/sort")
    public ApiResult<SortResult> sort(@RequestBody SortRequest request) {
        try {
            List<Integer> items = request.getItems();
            // R03 校验在 Service 层
            SortResult result = demoService.bubbleSort(items);
            return ApiResult.ok(result);
        } catch (IllegalArgumentException e) {
            log.warn("sort failed, illegal argument", e);
            return ApiResult.fail(e.getMessage());
        } catch (Exception e) {
            log.error("sort failed, unexpected error", e);
            return ApiResult.fail(DemoConstants.DEMO_001);
        }
    }

    /**
     * W04 导出。
     * POST /api/demo/export
     */
    @PostMapping("/export")
    public void export(@RequestBody ExportRequest request, HttpServletResponse response) {
        try {
            String tab = request.getTab();
            String format = request.getFormat();
            String[] result = exportService.export(tab, format);
            String contentType = result[0];
            String filename = result[1];
            String content = result[2];

            response.setContentType(contentType + ";charset=UTF-8");
            String encodedFilename = URLEncoder.encode(filename, StandardCharsets.UTF_8).replaceAll("\\+", "%20");
            response.setHeader(HttpHeaders.CONTENT_DISPOSITION, "attachment;filename=" + encodedFilename);
            response.setHeader("Access-Control-Expose-Headers", HttpHeaders.CONTENT_DISPOSITION);

            try (OutputStream os = response.getOutputStream()) {
                os.write(content.getBytes(StandardCharsets.UTF_8));
                os.flush();
            }
        } catch (IllegalArgumentException e) {
            log.warn("export failed, illegal argument", e);
            response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
            try (OutputStream os = response.getOutputStream()) {
                os.write(ApiResult.fail(e.getMessage() != null ? e.getMessage() : "export error")
                        .toString().getBytes(StandardCharsets.UTF_8));
            } catch (Exception ignored) {
                log.warn("write error response failed", ignored);
            }
        } catch (Exception e) {
            log.error("export failed, unexpected error", e);
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
            try (OutputStream os = response.getOutputStream()) {
                os.write(ApiResult.fail(DemoConstants.DEMO_001)
                        .toString().getBytes(StandardCharsets.UTF_8));
            } catch (Exception ignored) {
                log.warn("write error response failed", ignored);
            }
        }
    }
}

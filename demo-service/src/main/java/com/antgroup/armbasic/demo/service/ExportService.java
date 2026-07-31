package com.antgroup.armbasic.demo.service;

import com.antgroup.armbasic.demo.model.DemoConstants;
import com.antgroup.armbasic.demo.model.SortResult;
import lombok.extern.slf4j.Slf4j;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 导出服务（内部接口 S03）。
 * <p>
 * byte[] export(String tab, String format)
 * R04 tab 枚举校验；R05 format 不支持时回退 csv。
 * hello 导出文案，hash 导出原文+摘要，sort 导出原数组+排序结果+交换次数。
 * <p>
 * A8.3 改进：导出数据通过动态调用 DemoService 获取真实计算结果，不再使用硬编码样例字符串。
 */
@Slf4j
@Service
public class ExportService {

    @Autowired
    private DemoService demoService;

    /**
     * 导出指定 Tab 的展示结果。
     *
     * @param tab    hello/hash/sort
     * @param format csv/xlsx，默认 csv
     * @return [Content-Type, 文件名, 文件字节流]
     */
    public String[] export(String tab, String format) {
        // R04 tab 枚举校验
        if (!DemoConstants.inEnum(tab, DemoConstants.EXPORT_TABS)) {
            throw new IllegalArgumentException(DemoConstants.DEMO_005);
        }

        // R05 format 不支持时回退 csv
        String actualFormat = DemoConstants.DEFAULT_EXPORT_FORMAT;
        if (format != null && DemoConstants.inEnum(format, DemoConstants.EXPORT_FORMATS)) {
            actualFormat = format.toLowerCase();
        }

        byte[] content;
        String contentType;
        String extension;

        if ("xlsx".equals(actualFormat)) {
            content = exportXlsx(tab);
            contentType = DemoConstants.XLSX_CONTENT_TYPE;
            extension = "xlsx";
        } else {
            content = exportCsv(tab);
            contentType = DemoConstants.CSV_CONTENT_TYPE;
            extension = "csv";
        }

        String filename = "demo-" + tab + "." + extension;
        return new String[]{contentType, filename, new String(content, StandardCharsets.UTF_8)};
    }

    /**
     * CSV 导出。
     */
    private byte[] exportCsv(String tab) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        // 写 UTF-8 BOM 以兼容 Excel
        baos.write(0xEF);
        baos.write(0xBB);
        baos.write(0xBF);
        try (Writer writer = new OutputStreamWriter(baos, StandardCharsets.UTF_8)) {
            switch (tab) {
                case "hello":
                    writer.write("message\r\n");
                    writer.write(DemoConstants.HELLO_WORLD_MESSAGE + "\r\n");
                    break;
                case "sort": {
                    // A8.3: 动态调用 DemoService 获取真实排序结果
                    List<Integer> sampleItems = Arrays.stream(DemoConstants.EXPORT_SAMPLE_SORT_ITEMS)
                            .boxed().collect(Collectors.toList());
                    SortResult sortResult = demoService.bubbleSort(sampleItems);
                    writer.write("原数组,排序结果,交换次数\r\n");
                    writer.write(Arrays.toString(DemoConstants.EXPORT_SAMPLE_SORT_ITEMS) + ","
                            + sortResult.getSorted() + "," + sortResult.getSwapCount() + "\r\n");
                    break;
                }
                case "hash":
                default: {
                    // A8.3: 动态调用 DemoService 获取真实哈希结果
                    String[] hashResult = demoService.hash(
                            DemoConstants.EXPORT_SAMPLE_RAW, DemoConstants.EXPORT_SAMPLE_ALGORITHM);
                    writer.write("raw,algorithm,digest\r\n");
                    writer.write(DemoConstants.EXPORT_SAMPLE_RAW + "," + hashResult[0] + "," + hashResult[1] + "\r\n");
                    break;
                }
            }
            writer.flush();
        } catch (IOException e) {
            log.error("exportCsv failed, tab={}", tab, e);
            throw new RuntimeException(DemoConstants.DEMO_001, e);
        }
        return baos.toByteArray();
    }

    /**
     * XLSX 导出（Apache POI）。
     */
    private byte[] exportXlsx(String tab) {
        try (Workbook workbook = new XSSFWorkbook();
             ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            Sheet sheet = workbook.createSheet("demo-" + tab);
            switch (tab) {
                case "hello": {
                    Row header = sheet.createRow(0);
                    header.createCell(0).setCellValue("message");
                    Row data = sheet.createRow(1);
                    data.createCell(0).setCellValue(DemoConstants.HELLO_WORLD_MESSAGE);
                    break;
                }
                case "sort": {
                    // A8.3: 动态调用 DemoService 获取真实排序结果
                    List<Integer> sampleItems = Arrays.stream(DemoConstants.EXPORT_SAMPLE_SORT_ITEMS)
                            .boxed().collect(Collectors.toList());
                    SortResult sortResult = demoService.bubbleSort(sampleItems);
                    Row header = sheet.createRow(0);
                    header.createCell(0).setCellValue("原数组");
                    header.createCell(1).setCellValue("排序结果");
                    header.createCell(2).setCellValue("交换次数");
                    Row data = sheet.createRow(1);
                    data.createCell(0).setCellValue(Arrays.toString(DemoConstants.EXPORT_SAMPLE_SORT_ITEMS));
                    data.createCell(1).setCellValue(sortResult.getSorted().toString());
                    data.createCell(2).setCellValue(sortResult.getSwapCount());
                    break;
                }
                case "hash":
                default: {
                    // A8.3: 动态调用 DemoService 获取真实哈希结果
                    String[] hashResult = demoService.hash(
                            DemoConstants.EXPORT_SAMPLE_RAW, DemoConstants.EXPORT_SAMPLE_ALGORITHM);
                    Row header = sheet.createRow(0);
                    header.createCell(0).setCellValue("raw");
                    header.createCell(1).setCellValue("algorithm");
                    header.createCell(2).setCellValue("digest");
                    Row data = sheet.createRow(1);
                    data.createCell(0).setCellValue(DemoConstants.EXPORT_SAMPLE_RAW);
                    data.createCell(1).setCellValue(hashResult[0]);
                    data.createCell(2).setCellValue(hashResult[1]);
                    break;
                }
            }
            workbook.write(baos);
            return baos.toByteArray();
        } catch (IOException e) {
            // 降级为 CSV 流输出
            log.error("exportXlsx failed, tab={}", tab, e);
            throw new RuntimeException(DemoConstants.DEMO_001, e);
        }
    }
}

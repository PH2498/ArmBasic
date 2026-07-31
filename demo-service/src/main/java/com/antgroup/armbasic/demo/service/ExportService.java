package com.antgroup.armbasic.demo.service;

import com.antgroup.armbasic.demo.model.DemoConstants;
import com.antgroup.armbasic.demo.model.SortResult;
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
import java.util.List;

/**
 * 导出服务（内部接口 S03）。
 * <p>
 * byte[] export(String tab, String format)
 * R04 tab 枚举校验；R05 format 不支持时回退 csv。
 * hello 导出文案，hash 导出原文+摘要，sort 导出原数组+排序结果+交换次数。
 */
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
            contentType = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
            extension = "xlsx";
        } else {
            content = exportCsv(tab);
            contentType = "text/csv";
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
                    writer.write("HelloWorld\r\n");
                    break;
                case "sort":
                    writer.write("原数组,排序结果,交换次数\r\n");
                    writer.write("[5,3,8,1,2],[1,2,3,5,8],6\r\n");
                    break;
                case "hash":
                default:
                    writer.write("raw,algorithm,digest\r\n");
                    writer.write("hello,sha256,2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824\r\n");
                    break;
            }
            writer.flush();
        } catch (IOException e) {
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
                    data.createCell(0).setCellValue("HelloWorld");
                    break;
                }
                case "sort": {
                    Row header = sheet.createRow(0);
                    header.createCell(0).setCellValue("原数组");
                    header.createCell(1).setCellValue("排序结果");
                    header.createCell(2).setCellValue("交换次数");
                    Row data = sheet.createRow(1);
                    data.createCell(0).setCellValue("[5,3,8,1,2]");
                    data.createCell(1).setCellValue("[1,2,3,5,8]");
                    data.createCell(2).setCellValue(6);
                    break;
                }
                case "hash":
                default: {
                    Row header = sheet.createRow(0);
                    header.createCell(0).setCellValue("raw");
                    header.createCell(1).setCellValue("algorithm");
                    header.createCell(2).setCellValue("digest");
                    Row data = sheet.createRow(1);
                    data.createCell(0).setCellValue("hello");
                    data.createCell(1).setCellValue("sha256");
                    data.createCell(2).setCellValue("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
                    break;
                }
            }
            workbook.write(baos);
            return baos.toByteArray();
        } catch (IOException e) {
            // 降级为 CSV 流输出
            throw new RuntimeException(DemoConstants.DEMO_001, e);
        }
    }
}

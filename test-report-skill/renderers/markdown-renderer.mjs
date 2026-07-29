// Markdown 渲染器（M1 默认 / FR2 六章节固定顺序 / NFR4 幂等）
import { NA } from '../parsers/types.mjs';

/**
 * 将 TestRunResult 渲染为 Markdown 字符串。
 * generatedAt 在渲染期注入（满足 NFR4 幂等）。
 * @param {import('../parsers/types.mjs').TestRunResult} result
 * @param {{failThreshold?:number, coverageMode?:string}} [opts]
 * @returns {string}
 */
export function renderMarkdown(result, opts = {}) {
  const lines = [];
  const failThreshold = opts.failThreshold ?? null;
  const coverageMode = opts.coverageMode ?? 'auto';

  // ① 报告头
  lines.push('# 测试报告');
  lines.push('');
  lines.push('## 1. 报告头');
  lines.push('');
  lines.push(`- 项目名：${safe(result.header.projectName)}`);
  lines.push(`- 生成时间：${safe(result.header.generatedAt)}`);
  lines.push(`- 执行命令：${safe(result.header.command)}`);
  lines.push(`- 测试框架：${safe(result.header.framework)}`);
  lines.push(`- 框架版本：${safe(result.header.frameworkVersion)}`);
  lines.push(`- 执行环境：${safe(result.header.envSummary)}`);
  lines.push('');

  // ② 结果摘要
  const s = result.summary;
  const conclIcon = s.overall === 'passed' ? '✅' : '❌';
  const conclText = computeConclusionText(s, failThreshold);
  lines.push('## 2. 结果摘要');
  lines.push('');
  lines.push(`- 用例总数：${s.total}`);
  lines.push(`- 通过：${s.passed}`);
  lines.push(`- 失败：${s.failed}`);
  lines.push(`- 跳过：${s.skipped}`);
  lines.push(`- 通过率：${fmtPct(s.passRate)}`);
  lines.push(`- 总耗时：${fmtDuration(s.durationMs)}`);
  lines.push(`- 整体结论：${conclIcon} ${conclText}`);
  lines.push('');

  // ③ 失败用例分析（有失败时必选）
  if (s.failed > 0 && result.failures && result.failures.length) {
    lines.push('## 3. 失败用例分析');
    lines.push('');
    for (const f of result.failures) {
      lines.push(`### ${safe(f.name)}`);
      lines.push('');
      lines.push(`- 所属文件：${safe(f.filePath)}`);
      lines.push(`- 错误信息：${safe(f.errorMessage)}`);
      if (f.stackKeyLines && f.stackKeyLines.length) {
        lines.push('- 堆栈关键行：');
        lines.push('```');
        for (const l of f.stackKeyLines) lines.push(l);
        lines.push('```');
      }
      lines.push('');
    }
  }

  // ④ 用例明细
  lines.push(s.failed > 0 ? '## 4. 用例明细' : '## 3. 用例明细');
  lines.push('');
  if (result.details.groupedByFile.length === 0) {
    lines.push('> 未获取');
    lines.push('');
  } else {
    for (const g of result.details.groupedByFile) {
      lines.push(`### ${safe(g.filePath)}`);
      lines.push('');
      lines.push('| 用例 | 耗时 | 状态 |');
      lines.push('|---|---|---|');
      for (const c of g.cases) {
        lines.push(`| ${safe(c.name)} | ${fmtDuration(c.durationMs)} | ${statusIcon(c.status)} |`);
      }
      lines.push('');
    }
    if (result.details.truncated) {
      lines.push(`> ⚠️ ${result.details.noteWhenTruncated}`);
      lines.push('');
    }
  }

  // ⑤ 覆盖率
  const covIdx = s.failed > 0 ? 5 : 4;
  lines.push(`## ${covIdx}. 覆盖率`);
  lines.push('');
  if (coverageMode === 'off') {
    lines.push('> 已关闭覆盖率展示（coverage=off）');
    lines.push('');
  } else if (result.coverage) {
    lines.push('| 指标 | 覆盖率 |');
    lines.push('|---|---|');
    lines.push(`| 语句 | ${fmtPct(result.coverage.statements)} |`);
    lines.push(`| 分支 | ${fmtPct(result.coverage.branches)} |`);
    lines.push(`| 函数 | ${fmtPct(result.coverage.functions)} |`);
    lines.push(`| 行 | ${fmtPct(result.coverage.lines)} |`);
    lines.push('');
    if (result.coverage.belowThresholdFiles && result.coverage.belowThresholdFiles.length) {
      lines.push('**低于阈值的文件：**');
      lines.push('');
      lines.push('| 文件 | 指标 | 覆盖率 |');
      lines.push('|---|---|---|');
      for (const f of result.coverage.belowThresholdFiles) {
        lines.push(`| ${safe(f.filePath)} | ${safe(f.metric)} | ${fmtPct(f.value)} |`);
      }
      lines.push('');
    }
  } else {
    lines.push('> 未获取');
    lines.push('');
  }

  // ⑥ 附录
  const appIdx = covIdx + 1;
  lines.push(`## ${appIdx}. 附录`);
  lines.push('');
  lines.push(`- 原始结果文件：${safe(result.appendix.sourceResultPath)}`);
  lines.push(`- 生成工具：${safe(result.appendix.toolVersion)}`);
  lines.push('');

  return lines.join('\n');
}

function computeConclusionText(s, failThreshold) {
  if (s.failed > 0) return '存在失败用例';
  if (failThreshold != null && s.total > 0 && s.passRate < failThreshold) return `通过率低于阈值 ${failThreshold}%（不达标）`;
  return '全部通过';
}

function safe(v) { return v == null || v === '' ? NA : String(v); }
function fmtPct(v) { return typeof v === 'number' ? `${v.toFixed(1)}%` : safe(v); }
function fmtDuration(v) {
  if (v == null || v === NA || v === '') return NA;
  const n = Number(v);
  if (!Number.isFinite(n)) return String(v);
  if (n < 1000) return `${n}ms`;
  return `${(n / 1000).toFixed(2)}s`;
}
function statusIcon(s) {
  return s === 'passed' ? '✅ 通过' : s === 'failed' ? '❌ 失败' : '⏭️ 跳过';
}

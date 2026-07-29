// Vitest JSON reporter 解析器（spec FR1.2）
// 解析 vitest --reporter=json 的输出结构为 TestRunResult。
import {
  NA, TOOL_VERSION, computeOverall, passRateOf,
  truncateStack, DETAIL_LIMIT, UnparsableResultError,
} from './types.mjs';
import { sanitizeStack } from '../lib/sanitizer.mjs';
import { readFileSync } from 'node:fs';

export class VitestParser {
  get framework() { return 'vitest'; }

  canHandle(input) {
    const path = (input.resultFilePath || '').toLowerCase();
    if (path.endsWith('.json') && /vitest/i.test(path)) return true;
    if (input.rawOutput && looksLikeVitestJson(input.rawOutput)) return true;
    return false;
  }

  parse(input) {
    const raw = readInputContent(input);
    let data;
    try {
      data = JSON.parse(raw);
    } catch (e) {
      throw new UnparsableResultError(`Vitest JSON 解析失败：${e.message}`, { filePath: input.resultFilePath, position: 0 });
    }
    // vitest 结构：{ numTotalTests, numPassedTests, ..., testResults: [{ name, assertionResults }] }
    const testResults = data.testResults || (Array.isArray(data.files) ? data.files.map(f => ({ name: f.name || f.filepath, assertionResults: f.tests || f.tasks || [] })) : []);
    if (!testResults.length) {
      throw new UnparsableResultError('Vitest JSON 缺少 testResults/files 数据', { filePath: input.resultFilePath, position: 0 });
    }

    let total = 0, passed = 0, failed = 0, skipped = 0;
    const failures = [];
    const grouped = new Map();

    for (const file of testResults) {
      const filePath = file.name || file.filepath || NA;
      const grp = grouped.get(filePath) || { filePath, cases: [] };
      for (const t of (file.assertionResults || file.tasks || [])) {
        total++;
        const status = mapStatus(t.status || t.mode);
        if (status === 'passed') passed++;
        else if (status === 'failed') failed++;
        else if (status === 'skipped') skipped++;

        grp.cases.push({
          name: t.fullName || t.name || t.title || NA,
          durationMs: nonNaNum(t.duration) ? Math.round(t.duration) : NA,
          status,
        });

        if (status === 'failed') {
          const lines = collectFailureLines(t.errors || t.failureMessages);
          failures.push({
            name: t.fullName || t.name || t.title || NA,
            filePath,
            errorMessage: sanitizeStack(lines[0] || '未获取'),
            stackKeyLines: truncateStack(lines.map(sanitizeStack)),
          });
        }
      }
      grouped.set(filePath, grp);
    }

    // 优先用顶层聚合字段，否则用重计值
    total = nonNaNum(data.numTotalTests) ? data.numTotalTests : total;
    passed = nonNaNum(data.numPassedTests) ? data.numPassedTests : passed;
    failed = nonNaNum(data.numFailedTests) ? data.numFailedTests : failed;
    skipped = nonNaNum(data.numSkippedTests) ? data.numSkippedTests : skipped;

    const passRate = passRateOf(passed, total);
    const overall = computeOverall({ failed, total, passRate }, null);
    const groupedByFile = Array.from(grouped.values());
    const allCases = groupedByFile.reduce((n, g) => n + g.cases.length, 0);
    const truncated = allCases > DETAIL_LIMIT;

    return {
      header: {
        projectName: NA, generatedAt: NA,
        command: input.command || NA,
        framework: 'vitest', frameworkVersion: NA, envSummary: NA,
      },
      summary: { total, passed, failed, skipped, passRate, durationMs: nonNaNum(data.startTime) ? data.startTime : NA, overall },
      failures,
      details: {
        groupedByFile, truncated,
        noteWhenTruncated: truncated
          ? `用例明细超过 ${DETAIL_LIMIT} 条已截断展示（共 ${allCases} 条）`
          : NA,
      },
      appendix: { sourceResultPath: input.resultFilePath || NA, toolVersion: TOOL_VERSION },
    };
  }
}

function looksLikeVitestJson(raw) {
  try {
    const d = JSON.parse(raw);
    return !!(d && (Array.isArray(d.testResults) || Array.isArray(d.files) || 'numTotalTests' in d));
  } catch { return false; }
}

function readInputContent(input) {
  if (input.rawOutput && String(input.rawOutput).trim()) return String(input.rawOutput);
  if (input.resultFilePath) {
    const raw = readFileSync(input.resultFilePath, 'utf8');
    if (!raw || !raw.trim()) {
      throw new UnparsableResultError('Vitest JSON 文件为空', { filePath: input.resultFilePath, position: 0 });
    }
    return raw;
  }
  throw new UnparsableResultError('Vitest 解析器未获得任何输入内容', { filePath: input.resultFilePath, position: 0 });
}

function mapStatus(s) {
  if (s === 'passed' || s === 'run') return 'passed';
  if (s === 'failed' || s === 'fail' || s === 'errored') return 'failed';
  if (s === 'skipped' || s === 'todo' || s === 'skip' || s === 'only') return 'skipped';
  return s || 'failed';
}

function collectFailureLines(errors) {
  if (!Array.isArray(errors) || !errors.length) return [];
  const lines = [];
  for (const e of errors) {
    const msg = typeof e === 'string' ? e : (e?.message || e?.stack || '');
    if (!msg) continue;
    for (const line of String(msg).split('\n')) lines.push(line.trimEnd());
  }
  return lines.filter(Boolean);
}

function nonNaNum(v) { return typeof v === 'number' && Number.isFinite(v); }

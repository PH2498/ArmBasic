// pytest 解析器（M2 / NFR5：新增解析器不改既有解析器）
// 解析 pytest 的 JUnit XML report（复用 junit-xml.mjs）或 JSON report。
import {
  NA, TOOL_VERSION, computeOverall, passRateOf,
  truncateStack, DETAIL_LIMIT, UnparsableResultError,
} from './types.mjs';
import { sanitizeStack } from '../lib/sanitizer.mjs';
import { parseJunitXml } from '../lib/junit-xml.mjs';
import { readFileSync } from 'node:fs';

export class PytestParser {
  get framework() { return 'pytest'; }

  canHandle(input) {
    const path = (input.resultFilePath || '').toLowerCase();
    if (/pytest/i.test(path)) return true;
    if (input.rawOutput && looksLikePytestJson(input.rawOutput)) return true;
    return false;
  }

  parse(input) {
    const path = (input.resultFilePath || '').toLowerCase();
    // JSON report（pytest-json-report）优先
    if (path.endsWith('.json') || (input.rawOutput && looksLikePytestJson(input.rawOutput))) {
      return parseJson(input);
    }
    // 否则按 JUnit XML 解析，标注 framework=pytest
    const result = parseXml(input);
    result.header.framework = 'pytest';
    return result;
  }
}

function parseJson(input) {
  const raw = readInputContent(input);
  let data;
  try {
    data = JSON.parse(raw);
  } catch (e) {
    throw new UnparsableResultError(`pytest JSON 解析失败：${e.message}`, { filePath: input.resultFilePath, position: 0 });
  }
  const tests = data.tests || [];
  if (!Array.isArray(tests) || !tests.length) {
    throw new UnparsableResultError('pytest JSON 缺少 tests 数据', { filePath: input.resultFilePath, position: 0 });
  }

  let total = 0, passed = 0, failed = 0, skipped = 0;
  const failures = [];
  const grouped = new Map();

  for (const t of tests) {
    total++;
    const outcome = t.outcome || 'failed';
    const status = mapOutcome(outcome);
    if (status === 'passed') passed++;
    else if (status === 'failed') failed++;
    else if (status === 'skipped') skipped++;

    const filePath = t.filepath || t.file || NA;
    const grp = grouped.get(filePath) || { filePath, cases: [] };
    grp.cases.push({
      name: `${t.nodeid || t.name || NA}`,
      durationMs: nonNaNum(t.duration) ? Math.round(t.duration * 1000) : NA,
      status,
    });
    grouped.set(filePath, grp);

    if (status === 'failed') {
      const call = t.call || {};
      const lines = collectFailureLines(call.longrepr || call.crash?.message);
      failures.push({
        name: t.nodeid || t.name || NA,
        filePath,
        errorMessage: sanitizeStack(lines[0] || call.crash?.message || '未获取'),
        stackKeyLines: truncateStack(lines.map(sanitizeStack)),
      });
    }
  }

  // 顶层聚合优先
  total = nonNaNum(data.summary?.total) ? data.summary.total : total;
  passed = nonNaNum(data.summary?.passed) ? data.summary.passed : passed;
  failed = nonNaNum(data.summary?.failed) ? data.summary.failed : failed;
  skipped = nonNaNum(data.summary?.skipped) ? data.summary.skipped : skipped;

  const passRate = passRateOf(passed, total);
  const overall = computeOverall({ failed, total, passRate }, null);
  const groupedByFile = Array.from(grouped.values());
  const allCases = groupedByFile.reduce((n, g) => n + g.cases.length, 0);
  const truncated = allCases > DETAIL_LIMIT;

  return {
    header: {
      projectName: NA, generatedAt: NA,
      command: input.command || NA,
      framework: 'pytest',
      frameworkVersion: data.env?.Python || NA,
      envSummary: NA,
    },
    summary: { total, passed, failed, skipped, passRate, durationMs: nonNaNum(data.duration) ? Math.round(data.duration) : NA, overall },
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

function parseXml(input) {
  // 复用 JUnit XML 解析，复用 junit-parser 的核心逻辑但标注 pytest
  const content = readInputContent(input);
  const suites = parseJunitXml(content, { filePath: input.resultFilePath });
  let total = 0, passed = 0, failed = 0, skipped = 0, durationMs = 0;
  const failures = [];
  const grouped = new Map();
  for (const s of suites) {
    durationMs += nonNa(s.time) ? Math.round(s.time * 1000) : 0;
    for (const c of s.cases) {
      total++;
      const status = c.status;
      if (status === 'passed') passed++;
      else if (status === 'failed') failed++;
      else if (status === 'skipped') skipped++;
      const fileKey = c.classname || s.name || NA;
      const grp = grouped.get(fileKey) || { filePath: fileKey, cases: [] };
      grp.cases.push({ name: c.name || NA, durationMs: nonNa(c.time) ? Math.round(c.time * 1000) : NA, status });
      grouped.set(fileKey, grp);
      if (status === 'failed' && c.failure) {
        const lines = String(c.failure.message || '').split('\n').map((l) => l.trimEnd()).filter(Boolean);
        failures.push({
          name: c.name || NA, filePath: c.classname || s.name || NA,
          errorMessage: sanitizeStack(lines[0] || '未获取'),
          stackKeyLines: truncateStack(lines.map(sanitizeStack)),
        });
      }
    }
  }
  const passRate = passRateOf(passed, total);
  const overall = computeOverall({ failed, total, passRate }, null);
  return {
    header: { projectName: NA, generatedAt: NA, command: input.command || NA, framework: 'pytest', frameworkVersion: NA, envSummary: NA },
    summary: { total, passed, failed, skipped, passRate, durationMs: durationMs || NA, overall },
    failures,
    details: { groupedByFile: Array.from(grouped.values()), truncated: false, noteWhenTruncated: NA },
    appendix: { sourceResultPath: input.resultFilePath || NA, toolVersion: TOOL_VERSION },
  };
}

function looksLikePytestJson(raw) {
  try {
    const d = JSON.parse(raw);
    return !!(d && Array.isArray(d.tests) && (d.summary || d.created));
  } catch { return false; }
}

function readInputContent(input) {
  if (input.rawOutput && String(input.rawOutput).trim()) return String(input.rawOutput);
  if (input.resultFilePath) {
    const raw = readFileSync(input.resultFilePath, 'utf8');
    if (!raw || !raw.trim()) {
      throw new UnparsableResultError('pytest 结果文件为空', { filePath: input.resultFilePath, position: 0 });
    }
    return raw;
  }
  throw new UnparsableResultError('pytest 解析器未获得任何输入内容', { filePath: input.resultFilePath, position: 0 });
}

function mapOutcome(o) {
  if (o === 'passed') return 'passed';
  if (o === 'failed' || o === 'error') return 'failed';
  if (o === 'skipped' || o === 'xfailed' || o === 'xpassed') return 'skipped';
  return 'failed';
}

function collectFailureLines(longrepr) {
  if (!longrepr) return [];
  const text = typeof longrepr === 'string' ? longrepr : (longrepr?.message || longrepr?.repr || JSON.stringify(longrepr));
  return String(text).split('\n').map((l) => l.trimEnd()).filter(Boolean);
}

function nonNa(v) { return v != null && v !== NA; }
function nonNaNum(v) { return typeof v === 'number' && Number.isFinite(v); }

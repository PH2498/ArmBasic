// Jest JSON reporter 解析器（spec FR1.2 / Scenario 执行模式）
// 解析 `jest --json` 的 testResults 结构为 TestRunResult。
import {
  NA, TOOL_VERSION, computeOverall, passRateOf,
  truncateStack, DETAIL_LIMIT, UnparsableResultError,
} from './types.mjs';
import { sanitizeStack } from '../lib/sanitizer.mjs';
import { readFileSync } from 'node:fs';

export class JestParser {
  get framework() { return 'jest'; }

  canHandle(input) {
    const path = (input.resultFilePath || '').toLowerCase();
    if (path.endsWith('.json') && /jest/i.test(path)) return true;
    if (path === '' && input.rawOutput) {
      return looksLikeJestJson(input.rawOutput);
    }
    // 未命名 json：靠内容探测
    if (path.endsWith('.json') && input.rawOutput && looksLikeJestJson(input.rawOutput)) return true;
    return false;
  }

  parse(input) {
    const raw = readInputContent(input);
    let data;
    try {
      data = JSON.parse(raw);
    } catch (e) {
      throw new UnparsableResultError(`Jest JSON 解析失败：${e.message}`, { filePath: input.resultFilePath, position: 0 });
    }
    if (!data || !Array.isArray(data.testResults)) {
      throw new UnparsableResultError('Jest JSON 缺少 testResults 数组', { filePath: input.resultFilePath, position: 0 });
    }

    let total = 0, passed = 0, failed = 0, skipped = 0, durationMs = 0;
    const failures = [];
    const grouped = new Map();

    for (const file of data.testResults) {
      const filePath = file.name || NA;
      const grp = grouped.get(filePath) || { filePath, cases: [] };
      durationMs += nonNaNum(file.endTime) && nonNaNum(file.startTime)
        ? (file.endTime - file.startTime) : 0;

      for (const t of (file.assertionResults || [])) {
        total++;
        const status = mapStatus(t.status);
        if (status === 'passed') passed++;
        else if (status === 'failed') failed++;
        else if (status === 'skipped') skipped++;

        grp.cases.push({
          name: t.fullName || t.title || NA,
          durationMs: nonNaNum(t.duration) ? t.duration : NA,
          status,
        });

        if (status === 'failed') {
          const lines = collectFailureLines(t.failureMessages);
          failures.push({
            name: t.fullName || t.title || NA,
            filePath,
            errorMessage: sanitizeStack(lines[0] || t.failure?.[0]?.[0] || '未获取'),
            stackKeyLines: truncateStack(lines.map(sanitizeStack)),
          });
        }
      }
      grouped.set(filePath, grp);
    }

    const passRate = passRateOf(passed, total);
    const overall = computeOverall({ failed, total, passRate }, null);
    const groupedByFile = Array.from(grouped.values());
    const allCases = groupedByFile.reduce((n, g) => n + g.cases.length, 0);
    const truncated = allCases > DETAIL_LIMIT;

    return {
      header: {
        projectName: NA, generatedAt: NA,
        command: input.command || NA,
        framework: 'jest',
        frameworkVersion: data.testResults?.[0]?.version || data.jestVersion || NA,
        envSummary: NA,
      },
      summary: { total, passed, failed, skipped, passRate, durationMs: durationMs || NA, overall },
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

function looksLikeJestJson(raw) {
  try {
    const d = JSON.parse(raw);
    return !!(d && Array.isArray(d.testResults));
  } catch { return false; }
}

function readInputContent(input) {
  if (input.rawOutput && String(input.rawOutput).trim()) return String(input.rawOutput);
  if (input.resultFilePath) {
    const raw = readFileSync(input.resultFilePath, 'utf8');
    if (!raw || !raw.trim()) {
      throw new UnparsableResultError('Jest JSON 文件为空', { filePath: input.resultFilePath, position: 0 });
    }
    return raw;
  }
  throw new UnparsableResultError('Jest 解析器未获得任何输入内容', { filePath: input.resultFilePath, position: 0 });
}

function mapStatus(s) {
  if (s === 'passed') return 'passed';
  if (s === 'failed' || s === 'errored') return 'failed';
  if (s === 'pending' || s === 'skipped' || s === 'todo') return 'skipped';
  return s || 'failed';
}

function collectFailureLines(failureMessages) {
  if (!Array.isArray(failureMessages) || !failureMessages.length) return [];
  const lines = [];
  for (const item of failureMessages) {
    const msg = Array.isArray(item) ? item[0] : item;
    if (!msg) continue;
    for (const line of String(msg).split('\n')) lines.push(line.trimEnd());
  }
  return lines.filter(Boolean);
}

function nonNaNum(v) { return typeof v === 'number' && Number.isFinite(v); }

// JUnit XML 解析器（跨语言兜底，design §3 / spec FR1.2 解析模式 / NFR5）
import { parseJunitXml } from '../lib/junit-xml.mjs';
import {
  NA, TOOL_VERSION, computeOverall, passRateOf,
  truncateStack, DETAIL_LIMIT, UnparsableResultError,
} from './types.mjs';
import { sanitizeStack } from '../lib/sanitizer.mjs';
import { readFileSync } from 'node:fs';

export class JunitParser {
  get framework() { return 'junit'; }

  canHandle(input) {
    if (input.resultFilePath) {
      const lower = input.resultFilePath.toLowerCase();
      if (lower.endsWith('.xml') || /junit/i.test(lower)) return true;
    }
    if (input.rawOutput) {
      const t = String(input.rawOutput).trim();
      return /<testsuites|<testsuite/i.test(t);
    }
    return false;
  }

  parse(input) {
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
        grp.cases.push({
          name: c.name || NA,
          durationMs: nonNa(c.time) ? Math.round(c.time * 1000) : NA,
          status,
        });
        grouped.set(fileKey, grp);

        if (status === 'failed' && c.failure) {
          const lines = splitStackLines(c.failure.message);
          failures.push({
            name: c.name || NA,
            filePath: c.classname || s.name || NA,
            errorMessage: sanitizeStack(lines[0] || c.failure.message || '未获取'),
            stackKeyLines: truncateStack(lines.map(sanitizeStack)),
          });
        }
      }
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
        framework: 'junit', frameworkVersion: NA, envSummary: NA,
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

function readInputContent(input) {
  if (input.rawOutput && String(input.rawOutput).trim()) return String(input.rawOutput);
  if (input.resultFilePath) {
    const raw = readFileSync(input.resultFilePath, 'utf8');
    if (!raw || !raw.trim()) {
      throw new UnparsableResultError('JUnit XML 文件为空', { filePath: input.resultFilePath, position: 0 });
    }
    return raw;
  }
  throw new UnparsableResultError('JUnit 解析器未获得任何输入内容', { filePath: input.resultFilePath, position: 0 });
}

function nonNa(v) { return v != null && v !== NA; }

function splitStackLines(message) {
  if (!message) return [];
  return String(message).split('\n').map((l) => l.trimEnd()).filter(Boolean);
}

// 解析器契约与统一归一化数据模型（design.md §2/§3）
// 字段缺失统一用字符串 "未获取" 或省略可选段，不得崩溃或静默丢数据（NFR2）。

export const NA = '未获取';
export const TOOL_VERSION = 'test-report-skill 1.0.0';

/**
 * @typedef {Object} Failure
 * @property {string} name - 用例名
 * @property {string} filePath - 所属文件，缺失为 "未获取"
 * @property {string} errorMessage - 错误信息
 * @property {string[]} stackKeyLines - 堆栈关键行（默认 ≤10 行/≤2000 字符）
 */
/**
 * @typedef {Object} DetailCase
 * @property {string} name
 * @property {number|string} durationMs - 耗时，缺失为 "未获取"
 * @property {'passed'|'failed'|'skipped'} status
 */
/**
 * @typedef {Object} DetailGroup
 * @property {string} filePath
 * @property {DetailCase[]} cases
 */
/**
 * @typedef {Object} Coverage
 * @property {number|string} statements
 * @property {number|string} branches
 * @property {number|string} functions
 * @property {number|string} lines
 * @property {Array<{filePath:string,metric:string,value:number}>} belowThresholdFiles
 */
/**
 * @typedef {Object} TestRunResult
 * @property {{projectName:string,generatedAt:string,command:string,framework:string,frameworkVersion:string,envSummary:string}} header
 * @property {{total:number,passed:number,failed:number,skipped:number,passRate:number,durationMs:number|string,overall:'passed'|'failed'}} summary
 * @property {Failure[]} failures
 * @property {{groupedByFile:DetailGroup[],truncated:boolean,noteWhenTruncated:string}} details
 * @property {Coverage} [coverage]
 * @property {{sourceResultPath:string,toolVersion:string}} appendix
 */

/**
 * @typedef {Object} ParserInput
 * @property {'execute'|'parse'} mode
 * @property {string} [resultFilePath] - 解析模式必填
 * @property {string} [rawOutput] - 执行模式：命令 stdout/stderr 或落盘产物内容
 * @property {string} [coveragePath] - 可选覆盖率来源
 * @property {string} [command] - 执行命令（写入报告头）
 * @property {string} [cwd] - 工作目录（推断项目名）
 */

/**
 * 插件式解析器契约（NFR5）
 * @interface Parser
 * @property {string} framework - 框架标识 jest|vitest|junit|pytest
 * @property {(input:ParserInput)=>boolean} canHandle
 * @property {(input:ParserInput)=>TestRunResult} parse
 */

export class UnparsableResultError extends Error {
  constructor(message, { filePath, position } = {}) {
    super(message);
    this.name = 'UnparsableResultError';
    this.filePath = filePath;
    this.position = position;
  }
}

/**
 * 命令无法运行诊断（FR1.4）。非空报告冒充成功的反例。
 */
export class TestExecutionError extends Error {
  constructor(message, { exitCode, stderrSummary, suggestion } = {}) {
    super(message);
    this.name = 'TestExecutionError';
    this.exitCode = exitCode;
    this.stderrSummary = stderrSummary;
    this.suggestion = suggestion;
  }
}

// 生成空结果骨架（降级用，NFR2）
export function emptyResult(framework = 'unknown', extra = {}) {
  return {
    header: {
      projectName: NA,
      generatedAt: NA,
      command: NA,
      framework,
      frameworkVersion: NA,
      envSummary: NA,
      ...extra.header,
    },
    summary: {
      total: 0, passed: 0, failed: 0, skipped: 0,
      passRate: 0, durationMs: NA, overall: 'failed',
      ...extra.summary,
    },
    failures: extra.failures || [],
    details: { groupedByFile: [], truncated: false, noteWhenTruncated: NA, ...extra.details },
    appendix: { sourceResultPath: NA, toolVersion: TOOL_VERSION, ...extra.appendix },
  };
}

// 计算 overall：failed>0 或 passRate<fail_threshold（NFR2/FR4.2）
export function computeOverall({ failed, total, passRate }, failThreshold) {
  if (failed > 0) return 'failed';
  if (failThreshold != null && total > 0 && passRate < failThreshold) return 'failed';
  return 'passed';
}

// 通过率百分比，保留 1 位小数
export function passRateOf(passed, total) {
  if (!total) return 0;
  return Math.round((passed / total) * 1000) / 10;
}

// 堆栈关键行截断（默认 ≤10 行/≤2000 字符）
export function truncateStack(lines, maxLines = 10, maxChars = 2000) {
  if (!Array.isArray(lines)) return [];
  const trimmed = lines.filter(Boolean);
  const sliced = trimmed.slice(0, maxLines);
  let total = 0;
  const out = [];
  for (const l of sliced) {
    if (total + l.length > maxChars) {
      out.push(l.slice(0, Math.max(0, maxChars - total)) + ' …(已截断)');
      break;
    }
    out.push(l);
    total += l.length;
  }
  return out;
}

// 用例明细截断（>200 条，FR2）
export const DETAIL_LIMIT = 200;

// 调度层 + CLI 入口（design §1 三段式管道 / FR1.3 双模式 / FR3 落盘 / FR3.3 返回）
import { loadConfig, isParseMode } from './lib/config.mjs';
import { detect, detectProjectName } from './lib/detect.mjs';
import { defaultRegistry } from './parsers/registry.mjs';
import { renderMarkdown } from './renderers/markdown-renderer.mjs';
import { renderHtml } from './renderers/html-renderer.mjs';
import { renderJson } from './renderers/json-renderer.mjs';
import { buildEnvSummary } from './lib/env.mjs';
import { sanitize } from './lib/sanitizer.mjs';
import { TestExecutionError, NA } from './parsers/types.mjs';
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { spawnSync } from 'node:child_process';

/**
 * 主调度：解析配置 → 检测 → 解析结果 → 渲染 → 落盘 → 返回摘要。
 * @param {string[]} argv
 * @returns {Promise<{reportPath:string, summary:string, keyFailures?:string[]}>}
 */
export async function main(argv = process.argv.slice(2)) {
  const cfg = loadConfig(argv);
  const cwd = process.cwd();

  // 1. 检测框架/命令（执行模式需要；解析模式可从 result_file 推断框架）
  let detection = { framework: 'unknown', command: '', reporterArg: '', resultFile: cfg.result_file || '' };
  if (!isParseMode(cfg)) {
    detection = detect({ cwd, explicitCommand: cfg.test_command });
    if (!detection.command && !cfg.test_command) {
      throw new TestExecutionError('无法检测测试命令：未发现 package.json/pyproject.toml/Cargo.toml 或框架特征文件', {
        exitCode: null, stderrSummary: '', suggestion: '请显式指定 test_command，或在项目中配置测试脚本',
      });
    }
  }

  // 2. 构造 ParserInput（执行 or 解析）
  let parserInput;
  if (isParseMode(cfg)) {
    const rf = resolve(cfg.result_file);
    if (!existsSync(rf)) {
      throw new TestExecutionError(`解析模式：结果文件不存在 ${rf}`, { exitCode: null, stderrSummary: '', suggestion: '请确认 result_file 路径正确' });
    }
    parserInput = { mode: 'parse', resultFilePath: rf, command: cfg.test_command || NA };
  } else {
    // 执行模式：跑测试命令，收集产物
    const cmd = cfg.test_command || detection.command;
    const exec = runCommand(cmd, cwd);
    if (exec.error || (exec.status !== 0 && !hasAnyResult(exec.stdout))) {
      // 命令无法运行（非用例失败）→ 诊断，不生成空报告（FR1.4）
      throw new TestExecutionError(`测试命令无法运行：${exec.error?.message || 'exit ' + exec.status}`, {
        exitCode: exec.status,
        stderrSummary: sanitize((exec.stderr || '').slice(-500)),
        suggestion: '检查依赖是否安装、命令是否正确',
      });
    }
    // 收集落盘产物（优先 resultFile，否则用 stdout）
    const resultFile = detection.resultFile && existsSync(detection.resultFile) ? detection.resultFile : null;
    parserInput = {
      mode: 'execute',
      resultFilePath: resultFile,
      rawOutput: resultFile ? undefined : exec.stdout,
      command: cmd,
    };
  }

  // 3. 解析为 TestRunResult（插件匹配，JUnit 兜底）
  const result = defaultRegistry.parse(parserInput);

  // 4. 注入渲染期字段（NFR4 幂等：generatedAt 仅渲染期生成）
  result.header.projectName = detectProjectName(cwd);
  result.header.generatedAt = new Date().toISOString();
  result.header.envSummary = buildEnvSummary();
  if (parserInput.command) result.header.command = parserInput.command;
  result.appendix.sourceResultPath = parserInput.resultFilePath || result.appendix.sourceResultPath;

  // fail_threshold 影响 overall（FR4.2）
  if (cfg.fail_threshold != null) {
    result.summary.overall = (result.summary.failed > 0 ||
      (result.summary.total > 0 && result.summary.passRate < cfg.fail_threshold)) ? 'failed' : 'passed';
  }

  // 5. 覆盖率（M2：auto 探测 coveragePath）
  if (cfg.coverage !== 'off' && !result.coverage) {
    const cov = probeCoverage(cwd, detection.framework);
    if (cov) result.coverage = cov;
  }

  // 6. 渲染
  const renderOpts = { failThreshold: cfg.fail_threshold, coverageMode: cfg.coverage };
  let content;
  let ext;
  if (cfg.output_format === 'html') {
    content = renderHtml(result, renderOpts); ext = 'html';
  } else if (cfg.output_format === 'json') {
    content = renderJson(result); ext = 'json';
  } else {
    content = renderMarkdown(result, renderOpts); ext = 'md';
  }

  // 7. 落盘（FR3.2 默认 reports/test-report-<timestamp>.<ext>）
  const stamp = timestamp();
  const outDir = cfg.output_path || 'reports/';
  mkdirSync(outDir, { recursive: true });
  const reportPath = cfg.output_path && /\.\w+$/.test(cfg.output_path)
    ? cfg.output_path
    : join(outDir, `test-report-${stamp}.${ext}`);
  mkdirSync(dirname(reportPath), { recursive: true });
  writeFileSync(reportPath, content, 'utf8');

  // 8. 返回摘要 + 关键失败原因（FR3.3）
  const summaryStr = `通过率 ${result.summary.passRate}% | 通过 ${result.summary.passed}/${result.summary.total} | 失败 ${result.summary.failed} | 跳过 ${result.summary.skipped} | 结论 ${result.summary.overall === 'passed' ? '✅' : '❌'}`;
  const keyFailures = result.failures && result.failures.length
    ? result.failures.slice(0, 3).map((f) => `${f.name}：${f.errorMessage}`)
    : undefined;

  return { reportPath, summary: summaryStr, keyFailures, result };
}

function runCommand(cmd, cwd) {
  // 支持带参数的命令；用 shell 解析
  const res = spawnSync(cmd, [], { cwd, shell: true, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  return { status: res.status, stdout: res.stdout || '', stderr: res.stderr || '', error: res.error };
}

function hasAnyResult(stdout) {
  // stdout 含 JSON/XML 结构即视为有用例产出
  return /<testsuites|<testsuite|\{[\s\S]*testResults/.test(stdout || '');
}

function timestamp() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

// 覆盖率探测（M2）：auto 模式探测 coverage/coverage-final.json（Istanbul）等
function probeCoverage(cwd, framework) {
  const candidates = [
    join(cwd, 'coverage', 'coverage-summary.json'),
    join(cwd, 'coverage', 'coverage-final.json'),
    join(cwd, 'coverage.xml'),
  ];
  for (const p of candidates) {
    if (!existsSync(p)) continue;
    try {
      const raw = readFileSync(p, 'utf8');
      const data = JSON.parse(raw);
      return normalizeCoverage(data, p);
    } catch { /* 降级，忽略 */ }
  }
  return null;
}

function normalizeCoverage(data, path) {
  // Istanbul coverage-summary.json: { total: { lines: {pct}, ... } }
  const t = data.total || data;
  const pick = (obj, k) => (obj && typeof obj[k]?.pct === 'number') ? obj[k].pct : NA;
  const statements = pick(t, 'statements');
  const branches = pick(t, 'branches');
  const functions = pick(t, 'functions');
  const lines = pick(t, 'lines');
  if ([statements, branches, functions, lines].every((v) => v === NA)) return null;
  return { statements, branches, functions, lines, belowThresholdFiles: [] };
}

// CLI 入口
if (process.argv[1] && (process.argv[1].endsWith('index.mjs') || process.argv[1].endsWith('index.js'))) {
  main().then((out) => {
    console.log(`报告已生成：${out.reportPath}`);
    console.log(`摘要：${out.summary}`);
    if (out.keyFailures && out.keyFailures.length) {
      console.log('关键失败原因：');
      out.keyFailures.forEach((f) => console.log(`  - ${f}`));
    }
  }).catch((err) => {
    if (err.name === 'TestExecutionError') {
      console.error(`[测试执行错误] ${err.message}`);
      if (err.exitCode != null) console.error(`  exit code: ${err.exitCode}`);
      if (err.stderrSummary) console.error(`  stderr: ${err.stderrSummary}`);
      if (err.suggestion) console.error(`  建议: ${err.suggestion}`);
    } else if (err.name === 'UnparsableResultError') {
      console.error(`[解析错误] ${err.message}`);
      if (err.filePath) console.error(`  文件: ${err.filePath}`);
    } else {
      console.error(`[错误] ${err.stack || err.message}`);
    }
    process.exit(1);
  });
}

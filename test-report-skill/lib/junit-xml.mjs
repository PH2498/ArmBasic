// 轻量 JUnit XML 解析（无外部依赖，单次遍历，NFR1）
// 仅解析 testsuites/testsuite/testcase 结构，字段缺失降级（NFR2）。
import { UnparsableResultError, NA } from '../parsers/types.mjs';

/**
 * 解析 JUnit XML 字符串为中间结构（testsuite 列表）。
 * 不依赖 DOM，用正则提取节点；对损坏 XML 抛 UnparsableResultError。
 * @param {string} xml
 * @param {{filePath?:string}} [meta]
 */
export function parseJunitXml(xml, meta = {}) {
  if (typeof xml !== 'string' || !xml.trim()) {
    throw new UnparsableResultError('JUnit XML 为空或非字符串', { filePath: meta.filePath, position: 0 });
  }
  if (!/<testsuites|<testsuite/i.test(xml)) {
    throw new UnparsableResultError('未找到 testsuite/testsuites 节点，可能不是合法 JUnit XML', { filePath: meta.filePath, position: 0 });
  }

  const suites = [];
  // 提取每个 testsuite
  const suiteRe = /<testsuite\b([^>]*)\/?>([\s\S]*?)(?:<\/testsuite>|(?=<testsuite\b)|<\/testsuites>)/gi;
  const selfClosedSuiteRe = /<testsuite\b([^>]*)\/>/gi;

  let m;
  let matchedAny = false;
  // 自闭合 testsuite
  while ((m = selfClosedSuiteRe.exec(xml)) !== null) {
    matchedAny = true;
    suites.push({ attrs: m[1], body: '' });
  }
  // 带主体的 testsuite
  while ((m = suiteRe.exec(xml)) !== null) {
    // 跳过自闭合已处理的（body 为空且以 /> 结尾的 attrs）
    if (m[1].endsWith('/')) continue;
    matchedAny = true;
    suites.push({ attrs: m[1] || '', body: m[2] || '' });
  }
  if (!matchedAny) {
    throw new UnparsableResultError('解析 testsuite 失败：无法提取任何 testsuite 节点', { filePath: meta.filePath, position: 0 });
  }

  return suites.map((s) => parseSuite(s.attrs, s.body, meta));
}

function parseSuite(attrs, body, meta) {
  const a = parseAttrs(attrs);
  const cases = [];
  const caseRe = /<testcase\b([^>]*?)(?:\/>|>([\s\S]*?)<\/testcase>)/gi;
  let cm;
  while ((cm = caseRe.exec(body)) !== null) {
    const ca = parseAttrs(cm[1]);
    const inner = cm[2] || '';
    const failure = extractFailure(inner);
    cases.push({
      name: ca.name || NA,
      classname: ca.classname || NA,
      time: toNum(ca.time),
      status: failure ? 'failed' : (inner.includes('<skipped') ? 'skipped' : 'passed'),
      failure,
    });
  }
  return {
    name: a.name || NA,
    tests: toNum(a.tests),
    failures: toNum(a.failures),
    errors: toNum(a.errors),
    skipped: toNum(a.skipped),
    time: toNum(a.time),
    cases,
  };
}

function extractFailure(inner) {
  const m = inner.match(/<(failure|error)\b[^>]*>([\s\S]*?)<\/\1>/i);
  if (!m) return null;
  return { message: decodeXml(m[2] || ''), type: m[1].toLowerCase() };
}

function parseAttrs(s) {
  const out = {};
  if (!s) return out;
  const re = /(\w[\w-]*)\s*=\s*"([^"]*)"/g;
  let m;
  while ((m = re.exec(s)) !== null) out[m[1]] = m[2];
  return out;
}

function toNum(v) {
  if (v == null || v === '') return NA;
  const n = Number(v);
  return Number.isFinite(n) ? n : NA;
}

function decodeXml(s) {
  return s
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, '&');
}

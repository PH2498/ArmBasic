// 凭据脱敏（NFR3 / design §7）
// 正则脱敏 Bearer xxx / password= / token= / 私钥块；envSummary 不含环境变量与密钥。

const PATTERNS = [
  // Authorization: Bearer xxx
  { re: /(Bearer\s+)[A-Za-z0-9._\-=]+/gi, repl: '$1***' },
  // password=xxx / pwd=xxx / secret=xxx / token=xxx
  { re: /((?:password|passwd|pwd|secret|token|apikey|api_key|access_token|refresh_token)\s*[:=]\s*)[^\s&'"',;]+/gi, repl: '$1***' },
  // ?token=xxx / &password=xxx (URL/query)
  { re: /([?&](?:token|password|secret|api_key|apikey)=)[^&\s]+/gi, repl: '$1***' },
  // AWS keys (AKIA...)
  { re: /\b(AKIA|ASIA)[A-Z0-9]{12,}/g, repl: '***' },
  // 私钥块 (PEM)
  { re: /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g, repl: '*** [私钥块已脱敏] ***' },
  // 连接串中的密码 mongodb://user:pass@host -> user:***@
  { re: /([a-zA-Z][a-zA-Z0-9+.-]*:\/\/[^/@:]+:)[^@]+@/g, repl: '$1***@' },
];

/**
 * 对任意文本做凭据脱敏。
 * @param {string} text
 * @returns {string}
 */
export function sanitize(text) {
  if (typeof text !== 'string' || !text) return text;
  let out = text;
  for (const { re, repl } of PATTERNS) {
    out = out.replace(re, repl);
  }
  return out;
}

/**
 * 仅保留 node/python/os 版本等非敏感摘要（NFR3：不拷贝完整 env）。
 * @param {{node?:string,python?:string,os?:string,platform?:string}} info
 */
export function buildEnvSummary(info = {}) {
  const parts = [];
  if (info.node) parts.push(`node ${info.node}`);
  if (info.python) parts.push(`python ${info.python}`);
  if (info.os || info.platform) parts.push(`${info.os || info.platform}`);
  return parts.length ? parts.join(' · ') : '未获取';
}

// 对一段堆栈文本做脱敏（别名，保持语义清晰）
export function sanitizeStack(text) {
  return sanitize(text);
}

// 环境摘要（NFR3：仅保留 node/python/os 版本等非敏感信息，不拷贝完整 env）
import { sanitize } from './sanitizer.mjs';
import { execSync } from 'node:child_process';

/**
 * 构造报告头 envSummary。仅包含 node/python/os 版本，经脱敏兜底。
 * @returns {string}
 */
export function buildEnvSummary() {
  const parts = [];
  if (process.versions?.node) parts.push(`node ${process.versions.node}`);
  if (process.platform) parts.push(process.platform);
  if (process.arch) parts.push(process.arch);
  // python 可选探测（不阻塞，失败略）
  try {
    const py = execSync('python --version 2>&1 || python3 --version 2>&1', { encoding: 'utf8' }).trim();
    if (py) parts.push(py);
  } catch { /* python 不存在，略 */ }
  const summary = parts.length ? parts.join(' · ') : '未获取';
  return sanitize(summary);
}

// 配置项解析与默认值（FR4.2 / design §5）
import { resolve } from 'node:path';

export const DEFAULTS = {
  test_command: null,      // null = 自动检测
  result_file: null,       // null = 自动检测
  output_format: 'markdown',
  output_path: 'reports/',
  coverage: 'auto',        // auto / on / off
  fail_threshold: null,    // null = 无阈值；数值=百分比
};

const VALID_FORMATS = new Set(['markdown', 'html', 'json']);
const VALID_COVERAGE = new Set(['auto', 'on', 'off']);

/**
 * 解析命令行 / 用户覆盖项与默认值合并。
 * @param {string[]} argv - process.argv.slice(2)
 * @param {Object} [overrides] - 直接传入的覆盖对象
 */
export function loadConfig(argv = [], overrides = {}) {
  const cfg = { ...DEFAULTS };

  // 命令行：支持 --key value 与 --key=value 两种形式
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    let key, raw, consumedNext = false;
    const eq = a.indexOf('=');
    if (eq > 2) {
      key = a.slice(2, eq);
      raw = a.slice(eq + 1);
    } else {
      key = a.slice(2);
      raw = argv[i + 1];
      consumedNext = true;
    }
    if (raw == null || (consumedNext && raw.startsWith('--'))) continue; // 布尔开关略
    cfg[key] = coerce(key, raw);
    if (consumedNext) i++;
  }

  // 直接覆盖（优先级最高）
  for (const [k, v] of Object.entries(overrides)) {
    if (v == null) continue;
    cfg[k] = coerce(k, v);
  }

  return normalize(cfg);
}

function coerce(key, val) {
  switch (key) {
    case 'fail_threshold':
      return val === '' || val == null ? null : Number(val);
    case 'output_format':
    case 'coverage':
      return String(val).toLowerCase();
    default:
      return val;
  }
}

export function normalize(cfg) {
  if (!VALID_FORMATS.has(cfg.output_format)) cfg.output_format = DEFAULTS.output_format;
  if (!VALID_COVERAGE.has(cfg.coverage)) cfg.coverage = DEFAULTS.coverage;
  if (cfg.fail_threshold != null && Number.isNaN(cfg.fail_threshold)) cfg.fail_threshold = null;
  if (cfg.output_path) cfg.output_path = resolve(cfg.output_path);
  // 判断模式：有 result_file 或命令意图含文件 → 解析模式
  cfg.mode = cfg.result_file ? 'parse' : 'execute';
  return cfg;
}

// 是否跳过测试执行（解析模式）
export function isParseMode(cfg) {
  return cfg.mode === 'parse' || !!cfg.result_file;
}

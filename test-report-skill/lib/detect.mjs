// 框架/命令检测（FR1.1 / design §5）
// 优先级：用户显式指定 > 项目配置（package.json scripts.test / pyproject.toml / Cargo.toml）> 框架特征文件（jest.config.* / vitest.config.* / pytest.ini）
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { resolve, join, basename } from 'node:path';

/**
 * @typedef {Object} DetectResult
 * @property {string} framework - jest|vitest|pytest|junit|unknown
 * @property {string} command - 执行命令
 * @property {string} [reporterArg] - 生成 JSON/XML 产物的 reporter 参数
 * @property {string} [resultFile] - 预期结果文件路径
 */

/**
 * 检测测试框架与命令。
 * @param {Object} opts
 * @param {string} [opts.cwd]
 * @param {string} [opts.explicitCommand] - 用户显式指定
 * @param {string} [opts.explicitFramework]
 * @returns {DetectResult}
 */
export function detect({ cwd = process.cwd(), explicitCommand, explicitFramework } = {}) {
  // 1. 用户显式指定
  if (explicitCommand) {
    const fw = explicitFramework || inferFrameworkFromCommand(explicitCommand);
    return { framework: fw, command: explicitCommand, reporterArg: reporterArgFor(fw), resultFile: expectedResultFile(fw, cwd) };
  }

  // 2. 项目配置 / 特征文件
  const byPkg = detectFromPackageJson(cwd);
  if (byPkg) return byPkg;

  const byPyproject = detectFromPyproject(cwd);
  if (byPyproject) return byPyproject;

  const byCargo = detectFromCargo(cwd);
  if (byCargo) return byCargo;

  // 3. 特征文件推断
  const byFeature = detectFromFeatureFiles(cwd);
  if (byFeature) return byFeature;

  // 无法检测
  return { framework: 'unknown', command: '', reporterArg: '', resultFile: '' };
}

function inferFrameworkFromCommand(cmd) {
  const c = cmd.toLowerCase();
  if (c.includes('jest')) return 'jest';
  if (c.includes('vitest')) return 'vitest';
  if (c.includes('pytest')) return 'pytest';
  return 'unknown';
}

function reporterArgFor(framework) {
  switch (framework) {
    case 'jest': return '--json --outputFile=reports/jest-results.json';
    case 'vitest': return '--reporter=json --outputFile=reports/vitest-results.json';
    case 'pytest': return '--junitxml --junit-xml=reports/pytest-results.xml';
    default: return '';
  }
}

function expectedResultFile(framework, cwd) {
  switch (framework) {
    case 'jest': return join(cwd, 'reports/jest-results.json');
    case 'vitest': return join(cwd, 'reports/vitest-results.json');
    case 'pytest': return join(cwd, 'reports/pytest-results.xml');
    default: return '';
  }
}

function detectFromPackageJson(cwd) {
  const p = join(cwd, 'package.json');
  if (!existsSync(p)) return null;
  let pkg;
  try { pkg = JSON.parse(readFileSync(p, 'utf8')); } catch { return null; }

  // devDeps 指向框架
  const deps = { ...(pkg.devDependencies || {}), ...(pkg.dependencies || {}) };
  let framework = 'unknown';
  if (deps.vitest) framework = 'vitest';
  else if (deps.jest || deps['@jest/core']) framework = 'jest';

  const testScript = pkg.scripts?.test;
  if (testScript) {
    const fw = framework !== 'unknown' ? framework : inferFrameworkFromCommand(testScript);
    if (fw !== 'unknown') {
      return { framework: fw, command: `npm test -- ${reporterArgFor(fw)}`.trim(), reporterArg: reporterArgFor(fw), resultFile: expectedResultFile(fw, cwd) };
    }
  }

  if (framework !== 'unknown') {
    // 有依赖但无 test script
    const cmd = framework === 'vitest' ? 'npx vitest run' : 'npx jest';
    return { framework, command: `${cmd} ${reporterArgFor(framework)}`.trim(), reporterArg: reporterArgFor(framework), resultFile: expectedResultFile(framework, cwd) };
  }
  return null;
}

function detectFromPyproject(cwd) {
  const p = join(cwd, 'pyproject.toml');
  if (!existsSync(p)) return null;
  try {
    const txt = readFileSync(p, 'utf8');
    if (/\[tool\.pytest\]/.test(txt) || /pytest/.test(txt)) {
      return { framework: 'pytest', command: 'pytest ' + reporterArgFor('pytest'), reporterArg: reporterArgFor('pytest'), resultFile: expectedResultFile('pytest', cwd) };
    }
  } catch { /* ignore */ }
  return null;
}

function detectFromCargo(cwd) {
  const p = join(cwd, 'Cargo.toml');
  if (!existsSync(p)) return null;
  return { framework: 'unknown', command: 'cargo test', reporterArg: '', resultFile: '' };
}

function detectFromFeatureFiles(cwd) {
  let entries;
  try { entries = readdirSync(cwd); } catch { return null; }
  const has = (re) => entries.some((e) => re.test(e));
  if (has(/^jest\.config\./)) {
    return { framework: 'jest', command: 'npx jest ' + reporterArgFor('jest'), reporterArg: reporterArgFor('jest'), resultFile: expectedResultFile('jest', cwd) };
  }
  if (has(/^vitest\.config\./)) {
    return { framework: 'vitest', command: 'npx vitest run ' + reporterArgFor('vitest'), reporterArg: reporterArgFor('vitest'), resultFile: expectedResultFile('vitest', cwd) };
  }
  if (has(/^pytest\.ini$|^conftest\.py$/)) {
    return { framework: 'pytest', command: 'pytest ' + reporterArgFor('pytest'), reporterArg: reporterArgFor('pytest'), resultFile: expectedResultFile('pytest', cwd) };
  }
  return null;
}

// 项目名：取目录名（脱敏后）
export function detectProjectName(cwd = process.cwd()) {
  return basename(resolve(cwd)) || '未获取';
}

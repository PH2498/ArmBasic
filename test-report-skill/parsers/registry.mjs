// ParserRegistry：按 canHandle 匹配，JUnit 兜底最后（design §3 / spec FR1.4 / NFR5）
// 任何无法识别的输入回退 JUnit XML 解析器；再不行抛 UnparsableResultError（非空报告冒充成功）。
import { UnparsableResultError } from './types.mjs';
import { JunitParser } from './junit-parser.mjs';
import { JestParser } from './jest-parser.mjs';
import { VitestParser } from './vitest-parser.mjs';
import { PytestParser } from './pytest-parser.mjs';

/**
 * @typedef {import('./types.mjs').Parser} Parser
 * @typedef {import('./types.mjs').ParserInput} ParserInput
 * @typedef {import('./types.mjs').TestRunResult} TestRunResult
 */

// 注册顺序：具体框架在前，JUnit 兜底在最后。
// 新增框架 = 追加一个 Parser 实例到此数组，不修改既有解析器（NFR5）。
const PARSERS = [
  new JestParser(),
  new VitestParser(),
  new PytestParser(),
  new JunitParser(), // 兜底，永远放最后
];

export class ParserRegistry {
  constructor(parsers = PARSERS) {
    this.parsers = parsers;
    // 确保兜底解析器始终存在
    if (!this.parsers.some((p) => p.framework === 'junit')) {
      this.parsers.push(new JunitParser());
    }
  }

  /**
   * 按注册顺序找第一个 canHandle 的解析器；找不到回退 JUnit 兜底；
   * JUnit 兜底也无法解析则抛 UnparsableResultError。
   * @param {ParserInput} input
   * @returns {Parser}
   */
  resolve(input) {
    for (const p of this.parsers) {
      try {
        if (p.canHandle(input)) return p;
      } catch {
        // canHandle 异常不影响后续匹配
      }
    }
    // 兜底：junit-parser 总在末尾，它自己 canHandle 失败时 parse 会抛错
    const fallback = this.parsers[this.parsers.length - 1];
    return fallback;
  }

  /**
   * 解析为 TestRunResult。
   * @param {ParserInput} input
   * @returns {TestRunResult}
   */
  parse(input) {
    const parser = this.resolve(input);
    try {
      return parser.parse(input);
    } catch (err) {
      if (err.name === 'UnparsableResultError') throw err;
      // 包装为明确错误，不生成空报告冒充成功（FR1.4 / AC4）
      throw new UnparsableResultError(
        `解析失败：${err.message}`,
        { filePath: input.resultFilePath, position: 0 },
      );
    }
  }
}

export const defaultRegistry = new ParserRegistry();

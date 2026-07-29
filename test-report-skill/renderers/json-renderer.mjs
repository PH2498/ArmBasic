// JSON 渲染器（M3 伴随产物，FR3.1）输出结构化 TestRunResult JSON。
/**
 * @param {import('../parsers/types.mjs').TestRunResult} result
 * @returns {string}
 */
export function renderJson(result) {
  // stable stringify：键有序，保证幂等（NFR4）
  return JSON.stringify(result, Object.keys(result), 2);
}

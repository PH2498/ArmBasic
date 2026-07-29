// HTML 渲染器（M3，FR3.1，同六章节顺序）
import { renderMarkdown } from './markdown-renderer.mjs';
import { NA } from '../parsers/types.mjs';

/**
 * 将 TestRunResult 渲染为自包含 HTML 字符串。
 * 复用 Markdown 渲染逻辑后做最小 HTML 包装，保证章节顺序一致。
 * @param {import('../parsers/types.mjs').TestRunResult} result
 * @param {{failThreshold?:number, coverageMode?:string}} [opts]
 * @returns {string}
 */
export function renderHtml(result, opts = {}) {
  const md = renderMarkdown(result, opts);
  const title = `测试报告 - ${result.header.projectName || NA}`;
  const html = `<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(title)}</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222;line-height:1.6}
h1{border-bottom:2px solid #4a90d9;padding-bottom:.3rem}
h2{margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}
table{border-collapse:collapse;width:100%;margin:.5rem 0}
th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left}
th{background:#f5f7fa}
pre{background:#f6f8fa;padding:.8rem;overflow:auto;border-radius:4px}
code{background:#f6f8fa;padding:.1rem .3rem;border-radius:3px}
blockquote{color:#666;border-left:3px solid #ddd;padding-left:1rem;margin-left:0}
</style>
</head>
<body>
${mdToHtml(md)}
</body>
</html>`;
  return html;
}

// 最小 Markdown → HTML 转换（不引入外部依赖，覆盖报告所需语法）
function mdToHtml(md) {
  const lines = md.split('\n');
  let html = [];
  let inCode = false;
  let inTable = false;
  let listPending = false;
  for (const raw of lines) {
    const line = raw;
    if (line === '```') {
      if (inCode) { html.push('</code></pre>'); inCode = false; }
      else { html.push('<pre><code>'); inCode = true; }
      continue;
    }
    if (inCode) { html.push(escapeHtml(line)); continue; }
    if (line.startsWith('|') && line.includes('|')) {
      if (!inTable) { html.push('<table>'); inTable = true; }
      const cells = line.split('|').slice(1, -1).map((c) => c.trim());
      if (cells.every((c) => /^-+$/.test(c))) continue; // 分隔行
      const tag = /-{3,}/.test(cells.join('')) ? 'th' : 'td';
      const isHeader = false; // 简化：首行视作表头
      html.push('<tr>' + cells.map((c) => `<td>${inline(c)}</td>`).join('') + '</tr>');
      continue;
    } else if (inTable) {
      html.push('</table>'); inTable = false;
    }
    if (line.startsWith('# ')) { html.push(`<h1>${inline(line.slice(2))}</h1>`); continue; }
    if (line.startsWith('## ')) { html.push(`<h2>${inline(line.slice(3))}</h2>`); continue; }
    if (line.startsWith('### ')) { html.push(`<h3>${inline(line.slice(4))}</h3>`); continue; }
    if (line.startsWith('> ')) { html.push(`<blockquote>${inline(line.slice(2))}</blockquote>`); continue; }
    if (line.startsWith('- ')) { html.push(`<p>• ${inline(line.slice(2))}</p>`); continue; }
    if (line.trim() === '') { html.push(''); continue; }
    html.push(`<p>${inline(line)}</p>`);
  }
  if (inTable) html.push('</table>');
  if (inCode) html.push('</code></pre>');
  return html.join('\n');
}

function inline(s) {
  return escapeHtml(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

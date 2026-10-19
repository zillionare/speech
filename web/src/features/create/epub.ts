import { CallError } from '../../shared/lib/call-error';
import type { ManuscriptChapter } from './manuscript';

// EPUB 是 ZIP 包：按 OPF 的阅读顺序取出各个 XHTML 文档的正文，以一到三级标题切章。
const invalid = () => new CallError('@yovoice.book.invalid');

/** 读取 ZIP 中央目录，返回按需解压的条目表；只支持存储与 deflate 两种方式。 */
function unzip(bytes: Uint8Array): Map<string, () => Promise<Uint8Array>> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let end = -1;
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 65557); i--) if (view.getUint32(i, true) === 0x06054b50) { end = i; break; }
  if (end < 0) throw invalid();
  const entries = new Map<string, () => Promise<Uint8Array>>();
  let offset = view.getUint32(end + 16, true);
  const count = view.getUint16(end + 10, true);
  const decoder = new TextDecoder();
  for (let n = 0; n < count; n++) {
    if (offset + 46 > bytes.length || view.getUint32(offset, true) !== 0x02014b50) throw invalid();
    const method = view.getUint16(offset + 10, true), size = view.getUint32(offset + 20, true);
    const nameLength = view.getUint16(offset + 28, true), extra = view.getUint16(offset + 30, true), comment = view.getUint16(offset + 32, true);
    const local = view.getUint32(offset + 42, true);
    const name = decoder.decode(bytes.subarray(offset + 46, offset + 46 + nameLength));
    entries.set(name, async () => {
      if (local + 30 > bytes.length || view.getUint32(local, true) !== 0x04034b50) throw invalid();
      const start = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true);
      const data = bytes.subarray(start, start + size);
      if (method === 0) return data;
      if (method !== 8) throw invalid();
      const stream = new Blob([data as BlobPart]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
      return new Uint8Array(await new Response(stream).arrayBuffer());
    });
    offset += 46 + nameLength + extra + comment;
  }
  return entries;
}

function parseXML(text: string, type: DOMParserSupportedType = 'application/xml') {
  const doc = new DOMParser().parseFromString(text, type);
  return doc.getElementsByTagName('parsererror').length ? new DOMParser().parseFromString(text, 'text/html') : doc;
}

const resolve = (base: string, href: string) => {
  const parts = (base.includes('/') ? base.slice(0, base.lastIndexOf('/') + 1) : '').concat(decodeURIComponent(href.split('#')[0])).split('/');
  const out: string[] = [];
  for (const part of parts) { if (part === '..') out.pop(); else if (part && part !== '.') out.push(part); }
  return out.join('/');
};

const blocks = new Set(['p', 'div', 'li', 'blockquote', 'section', 'article', 'br', 'tr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre']);

/** 正文转成按段换行的纯文本；一到三级标题单独返回，用来切章。 */
function documentChapters(doc: Document, fallback: string): ManuscriptChapter[] {
  const chapters: ManuscriptChapter[] = [{ title: fallback, text: '' }];
  let line = '';
  const flush = () => { if (line.trim()) chapters.at(-1)!.text += line.trim() + '\n'; line = ''; };
  const walk = (node: Node) => {
    if (node.nodeType === Node.TEXT_NODE) { line += node.textContent?.replace(/\s+/g, ' ') ?? ''; return; }
    if (node.nodeType !== Node.ELEMENT_NODE) return;
    const tag = (node as Element).localName.toLowerCase();
    if (['script', 'style', 'head', 'rt', 'rp', 'sup'].includes(tag)) return;
    if (/^h[1-3]$/.test(tag)) {
      flush();
      const title = node.textContent?.replace(/\s+/g, ' ').trim() ?? '';
      if (title && chapters.at(-1)!.text.trim()) chapters.push({ title, text: '' });
      else if (title) chapters.at(-1)!.title = title;
      return;
    }
    if (blocks.has(tag)) flush();
    node.childNodes.forEach(walk);
    if (blocks.has(tag)) flush();
  };
  walk(doc.body ?? doc.documentElement);
  flush();
  return chapters.filter(c => c.text.trim());
}

export async function readEpub(bytes: Uint8Array): Promise<{ title: string; chapters: ManuscriptChapter[] }> {
  if (typeof DecompressionStream === 'undefined') throw invalid();
  const entries = unzip(bytes);
  const text = async (name: string) => { const read = entries.get(name); if (!read) throw invalid(); return new TextDecoder().decode(await read()); };
  const container = parseXML(await text('META-INF/container.xml'));
  const opfPath = container.getElementsByTagName('rootfile')[0]?.getAttribute('full-path');
  if (!opfPath) throw invalid();
  const opf = parseXML(await text(opfPath));
  const title = opf.getElementsByTagName('dc:title')[0]?.textContent?.trim() || opf.getElementsByTagNameNS('*', 'title')[0]?.textContent?.trim() || '';
  const manifest = new Map([...opf.getElementsByTagNameNS('*', 'item')].map(item => [item.getAttribute('id') ?? '', item]));
  const chapters: ManuscriptChapter[] = [];
  for (const ref of opf.getElementsByTagNameNS('*', 'itemref')) {
    if (ref.getAttribute('linear') === 'no') continue;
    const item = manifest.get(ref.getAttribute('idref') ?? '');
    if (!item || /\bnav\b/.test(item.getAttribute('properties') ?? '') || !/html/.test(item.getAttribute('media-type') ?? '')) continue;
    const path = resolve(opfPath, item.getAttribute('href') ?? '');
    if (!entries.has(path)) continue;
    const source = await text(path);
    const doc = parseXML(source, 'application/xhtml+xml');
    // 文档标题常常就是书名，这时不当作章名。
    const fallback = doc.getElementsByTagName('title')[0]?.textContent?.trim() || '';
    chapters.push(...documentChapters(doc, fallback === title ? '' : fallback));
  }
  return { title, chapters };
}

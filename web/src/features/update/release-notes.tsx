import { VStack } from '@astryxdesign/core/Layout';
import type { UiLocale } from '../../shared/workbench';

export interface NoteBlock { kind: 'item' | 'text'; text: string }

// 发布说明来自 CHANGELOG：版本标题下分 ### English 与 ### 简体中文，各自是要点列表。
// 中文界面取简体中文一节，其余语言取英文；缺少该节时退回另一种语言，再退回整段正文。
export function releaseNotes(body: string, locale: UiLocale): NoteBlock[] {
  const text = body.replace(/\r\n?/g, '\n');
  const sections = new Map<string, string>();
  for (const match of text.matchAll(/^###\s+(English|简体中文)\s*\n([\s\S]*?)(?=^#{1,3}\s|(?![\s\S]))/gm)) sections.set(match[1], match[2]);
  const preferred = locale.startsWith('zh') ? ['简体中文', 'English'] : ['English', '简体中文'];
  const section = preferred.map(name => sections.get(name)).find(value => value?.trim()) ?? text;
  return section.split('\n').map(line => line.trim())
    .filter(line => line && !line.startsWith('#') && !line.startsWith('<!--'))
    .map(line => /^[-*]\s+/.test(line) ? { kind: 'item' as const, text: line.replace(/^[-*]\s+/, '') } : { kind: 'text' as const, text: line });
}

// 只渲染行内代码，其余 Markdown 标记去掉，不解析 HTML。
function Inline({ text }: { text: string }) {
  return <>{text.replace(/\*\*(.+?)\*\*/g, '$1').split(/(`[^`]+`)/).map((part, index) => part.startsWith('`') && part.endsWith('`') && part.length > 2 ? <code key={index}>{part.slice(1, -1)}</code> : part)}</>;
}

export function ReleaseNotes({ body, locale }: { body: string; locale: UiLocale }) {
  // 连续的要点合成一个列表，其余行按原顺序显示为段落。
  const groups: NoteBlock[][] = [];
  for (const block of releaseNotes(body, locale)) {
    const last = groups.at(-1);
    if (block.kind === 'item' && last?.[0].kind === 'item') last.push(block); else groups.push([block]);
  }
  return <VStack className="release-notes" gap={2} data-testid="release-notes">
    {groups.map((group, index) => group[0].kind === 'item'
      ? <ul key={index}>{group.map((block, item) => <li key={item}><Inline text={block.text} /></li>)}</ul>
      : <p key={index}><Inline text={group[0].text} /></p>)}
  </VStack>;
}

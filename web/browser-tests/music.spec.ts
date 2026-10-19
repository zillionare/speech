import { test, expect } from '@playwright/test';
import { seedState } from './seed';
import { emptyState } from '../src/shared/workbench';
import { createMusicDraft } from '../src/features/music/music-draft';
import { songLyrics, syntheticSong, wavBytes } from './synthetic-song';

test('新建音乐作品：示例、风格标签、段落标记、纯音乐与参数持久化', async ({ page }, testInfo) => {
  await page.goto('/');
  await page.getByTestId('nav-new').click();
  await page.getByRole('menuitem', { name: '音乐生成', exact: true }).click();
  const style = page.getByLabel('风格', { exact: true });
  await expect(style).toHaveValue('');
  await page.getByRole('button', { name: '填入示例', exact: true }).click();
  await expect(style).toHaveValue(/city pop/);
  await expect(page.getByRole('button', { name: '填入示例', exact: true })).toHaveCount(0);
  // 风格预设与微调元素位于右侧检查器（纯文字无 emoji，与情绪预设一致）。
  await expect(page.getByRole('button', { name: 'City Pop', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'City Pop', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.screenshot({ path: testInfo.outputPath('music-style-presets-top.png') });

  // 点击微调元素展开分类标签。
  await page.getByRole('button', { name: '微调元素', exact: true }).click();
  const tags = page.getByRole('group', { name: '风格标签' });
  await expect(tags.getByRole('button', { name: 'city pop', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await tags.getByRole('button', { name: 'piano', exact: true }).click();
  await expect(style).toHaveValue(/, piano$/);
  await tags.getByRole('button', { name: 'city pop', exact: true }).click();
  await expect(style).not.toHaveValue(/city pop/);
  await page.screenshot({ path: testInfo.outputPath('music-style-presets.png') });

  const lyrics = page.getByLabel('歌词', { exact: true });
  await lyrics.evaluate((el: HTMLTextAreaElement) => el.setSelectionRange(el.value.length, el.value.length));
  await page.getByRole('button', { name: '插入段落' }).click();
  await page.getByRole('menuitem', { name: '[Bridge]', exact: true }).click();
  await expect(lyrics).toHaveValue(/一直等\n\n\[Bridge\]\n$/);

  // 音乐作品使用独立检查器：只列音乐模型，时长与语言写入作品。
  const model = page.getByRole('combobox', { name: '模型' });
  await expect(model).toContainText('ACE-Step 1.5 Turbo');
  await page.getByRole('radio', { name: '02:00' }).click();
  await page.getByRole('combobox', { name: '演唱语言' }).click();
  await page.getByRole('option', { name: '粤语' }).click();
  await page.getByRole('radio', { name: '纯音乐' }).click();
  await expect(lyrics).toHaveCount(0);
  await page.screenshot({ path: testInfo.outputPath('music.png') });

  await expect(page.locator('.saved')).toContainText('已保存');
  await page.reload();
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('voice-workbench-v1')!).drafts.find((d: { kind?: string }) => d.kind === 'music'));
  expect(saved).toMatchObject({ kind: 'music', modelId: 'ace-step-1.5-turbo-bf16', instrumental: true, synthesisLanguage: 'yue', modelOptions: { ace_step: { duration_seconds: 120 } } });
  expect(saved.lyrics).toContain('[Bridge]');
  await expect(page.getByRole('radio', { name: '纯音乐' })).toBeChecked();

  // 未安装模型时生成直接打开“音乐模型”设置；配音作品的模型列表不包含音乐模型。
  await page.getByRole('button', { name: '生成歌曲', exact: true }).click();
  await expect(page.getByRole('tab', { name: '音乐模型', selected: true })).toBeVisible();
  await page.getByTestId('nav-music').click();
  await expect(page.locator('.project-library-row:visible .record-cover')).toHaveCount(1);
  await page.getByTestId('nav-new').click();
  await page.getByRole('menuitem', { name: '语音生成', exact: true }).click();
  await page.getByRole('combobox', { name: '模型' }).click();
  await expect(page.getByRole('option', { name: /ACE-Step/ })).toHaveCount(0);
});


test('选中 YuE2：长度由歌词决定，纯音乐先写谱再交给乐器，可选生成方式与高级参数', async ({ page }, testInfo) => {
  const draft = { ...createMusicDraft('zh-CN'), id: 'd'.repeat(32), modelId: 'ace-step-1.5-turbo-q8', instrumental: true };
  const state = emptyState();
  state.drafts = [draft];
  state.models = [
    { id: 'ace-step-1.5-turbo-q8', path: 'ace-step-1.5-turbo-q8_0.gguf', managed: true },
    { id: 'yue2-3b-q8', path: 'Yue2-3B-GGUF', managed: true },
  ];
  await page.addInitScript(value => { if (!localStorage.getItem('voice-workbench-v1')) localStorage.setItem('voice-workbench-v1', value); }, JSON.stringify(state));
  await page.goto('/');
  await expect(page.getByRole('radio', { name: '纯音乐' })).toBeChecked();
  await page.getByRole('combobox', { name: '模型' }).click();
  await page.getByRole('option', { name: /^YuE2 3B/ }).click();
  // YuE2 也能生成纯音乐：保留纯音乐，歌词变为可选的段落结构，生成方式固定按谱；ACE-Step 专属参数隐藏。
  await expect(page.getByRole('radio', { name: '纯音乐' })).toBeChecked();
  await expect(page.getByLabel('歌词', { exact: true })).toHaveAttribute('placeholder', /段落标记/);
  await expect(page.getByRole('radio', { name: '02:00' })).toHaveCount(0);
  await expect(page.getByRole('combobox', { name: '演唱语言' })).toHaveCount(0);
  await expect(page.getByRole('radio', { name: '自由发挥' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: '先写旋律谱' })).toBeEnabled();
  await page.getByLabel('歌词', { exact: true }).fill('');
  await expect(page.getByRole('button', { name: '生成歌曲', exact: true })).toBeEnabled();
  await page.screenshot({ path: testInfo.outputPath('music-yue2-instrumental.png') });
  // 填写歌词时可以选不经过谱自由发挥；高级设置里是 YuE2 自己的参数。
  await page.getByRole('radio', { name: '填写歌词' }).click();
  await expect(page.getByRole('button', { name: '生成歌曲', exact: true })).toBeDisabled();
  await expect(page.getByRole('radio', { name: '按旋律谱' })).toBeChecked();
  await page.getByRole('radio', { name: '自由发挥' }).click();
  await page.locator('details.advanced > summary').click();
  await expect(page.getByLabel('演唱温度')).toHaveValue('1');
  await expect(page.getByLabel('写谱温度')).toHaveValue('0.7');
  await expect(page.getByLabel('时长')).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => JSON.parse(localStorage.getItem('voice-workbench-v1') ?? '{}').drafts?.[0]?.modelOptions?.yue2?.cot)).toBe('off');
  await page.screenshot({ path: testInfo.outputPath('music-yue2.png') });
});

const melodyABC = `X:1
T:
M:4/4
L:1/32
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% verse
V: Vocal
"C"E8G8c16|"G"B8G8D16|
V: Ins
Z2|
`;

test('YuE2 旋律谱：导入 ABC，看轨道与谱面，调速度、只留旋律后按谱生成', async ({ page }, testInfo) => {
  const draft = { ...createMusicDraft('zh-CN'), id: 'f'.repeat(32), modelId: 'yue2-3b-q8', instrumental: false };
  const state = emptyState();
  state.drafts = [draft];
  state.models = [{ id: 'yue2-3b-q8', path: 'Yue2-3B-GGUF', managed: true }];
  await seedState(page, state);
  await page.goto('/');
  // 空状态给出三种来源：按歌词出谱、从歌曲提取、导入谱文件。
  await expect(page.getByRole('heading', { name: '旋律谱' })).toBeVisible();
  await expect(page.getByRole('button', { name: '先写旋律谱' })).toBeEnabled();
  await expect(page.getByRole('button', { name: '生成歌曲', exact: true })).toBeVisible();
  await page.getByLabel('ABC 谱文件').setInputFiles({ name: 'song.abc', mimeType: 'text/plain', buffer: Buffer.from(melodyABC) });
  await expect(page.getByText(/C · 4\/4 · 2 小节 · 00:04 · 带和弦/)).toBeVisible();
  // 默认用五线谱显示，可以切到轨道或 ABC 谱面。
  await expect(page.getByRole('img', { name: '五线谱' }).locator('svg')).toBeVisible();
  // 试听要用编曲音色库渲染；没装音色库时提示去下载。
  await page.getByRole('button', { name: '试听', exact: true }).click();
  await expect(page.getByText('试听旋律谱需要一个音色库')).toBeVisible();
  await page.getByRole('radio', { name: '轨道' }).click();
  await expect(page.getByRole('group', { name: '旋律谱轨道，共 2 小节' })).toBeVisible();
  await expect(page.getByRole('button', { name: '按旋律谱生成', exact: true })).toBeVisible();
  await page.getByRole('group', { name: '旋律谱轨道，共 2 小节' }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: testInfo.outputPath('melody-tracks.png') });

  const tempo = page.getByRole('spinbutton', { name: '速度' });
  await tempo.fill('90'); await tempo.press('Enter');
  await expect(page.getByText(/2 小节 · 00:05/)).toBeVisible();
  await expect(tempo).toHaveValue('90');
  await page.getByRole('button', { name: '谱面操作' }).click();
  await page.getByRole('menuitem', { name: '只保留旋律' }).click();
  await expect(page.getByText(/仅旋律/)).toBeVisible();
  // 谱面改动可以逐步撤回：先撤回只保留旋律，再撤回改速度，然后重新做一遍。
  await page.getByRole('button', { name: '撤回：只保留旋律' }).click();
  await expect(page.getByText(/带和弦/)).toBeVisible();
  await page.getByRole('button', { name: '撤回：速度' }).click();
  await expect(tempo).toHaveValue('120');
  await expect(page.getByRole('button', { name: /^撤回/ })).toHaveAccessibleName('撤回：导入 .abc');
  await tempo.fill('90'); await tempo.press('Enter');
  await expect(tempo).toHaveValue('90');
  await page.getByRole('button', { name: '谱面操作' }).click();
  await page.getByRole('menuitem', { name: '只保留旋律' }).click();
  await expect(page.getByText(/仅旋律/)).toBeVisible();
  // 导出为 .abc 文件。
  await page.getByRole('button', { name: '谱面操作' }).click();
  const exported = page.waitForEvent('download');
  await page.getByRole('menuitem', { name: '导出 .abc' }).click();
  expect((await exported).suggestedFilename()).toMatch(/\.abc$/);
  await page.getByRole('radio', { name: 'ABC 谱面' }).click();
  const source = page.getByLabel('ABC 谱面', { exact: true });
  await expect(source).toHaveValue(/Q:1\/4=90/);
  await expect(source).not.toHaveValue(/"C"/);
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('voice-workbench-v1') ?? '{}').drafts[0].abc as string);
  expect(saved).toContain('Q:1/4=90');
  await source.scrollIntoViewIfNeeded();
  await page.screenshot({ path: testInfo.outputPath('melody-source.png') });

  // 浏览器预览不做推理：提取旋律给出明确提示。
  await page.getByRole('button', { name: '谱面操作' }).click();
  await page.getByRole('menuitem', { name: '清空谱' }).click();
  // 清空也能撤回。
  await page.getByRole('button', { name: '撤回：清空谱' }).click();
  await expect(page.getByText(/仅旋律/)).toBeVisible();
  await page.getByRole('button', { name: '谱面操作' }).click();
  await page.getByRole('menuitem', { name: '清空谱' }).click();
  await page.getByLabel('要提取旋律的歌曲').setInputFiles({ name: 'song.wav', mimeType: 'audio/wav', buffer: Buffer.from(wavBytes(new Float32Array(800), new Float32Array(800), 8000)) });
  await expect(page.getByRole('alert').filter({ hasText: '桌面' })).toBeVisible();
});

test('音乐历史版本：工具栏进入历史页，播放页、复用和删除都在这里', async ({ page }) => {
  const draft = { ...createMusicDraft('zh-CN'), id: 'e'.repeat(32) };
  const state = emptyState();
  state.drafts = [draft];
  const take = (id: string, hour: number, lyrics: string) => ({ id, title: draft.title, fileName: `${id}.wav`, createdAt: `2026-09-15T${String(hour).padStart(2, '0')}:00:00Z`, duration: 30, settings: { ...draft, text: `style ${id}`, lyrics } });
  state.history = [take('v3', 12, '[verse]\n第三版的歌词'), take('v2', 11, '[verse]\n第二版的第一句\n第二版的第二句'), take('v1', 10, '[verse]\n第一版的歌词')];
  await page.addInitScript(value => { if (!localStorage.getItem('voice-workbench-v1')) localStorage.setItem('voice-workbench-v1', value); }, JSON.stringify(state));
  await page.goto('/');
  // 编辑区不再有版本列表；播放器工具栏只放导出、播放页和历史版本入口。
  await expect(page.locator('.music-take-row')).toHaveCount(0);
  const transport = page.locator('.player .transport');
  await expect(transport.getByRole('button', { name: '导出音频', exact: true })).toBeVisible();
  await expect(transport.getByRole('button', { name: '播放页', exact: true })).toBeVisible();
  await transport.getByRole('button', { name: '历史版本', exact: true }).click();
  await expect(page.getByRole('heading', { name: `${draft.title} · 历史版本` })).toBeVisible();
  const rows = page.locator('.history-item');
  await expect(rows).toHaveCount(3);
  // 历史版本用唱片票根：唱片中心试听，标题是风格，摘录是第一句歌词，存根是时长。
  await expect(rows.locator('.record-cover')).toHaveCount(3);
  await expect(rows.locator('.history-meta b')).toHaveText(['版本 3', '版本 2', '版本 1']);
  await expect(rows.nth(1).locator('.project-excerpt')).toHaveText('「第二版的第一句」');
  await expect(rows.nth(1).locator('.ticket-stamp')).toContainText('0:30');

  // 播放页：滚动歌词按段落估算时间，器乐间隔显示为省略号，关闭后回到历史页。
  await rows.nth(1).getByRole('button', { name: '在播放页打开 style v2', exact: true }).click();
  const playing = page.getByRole('dialog', { name: `${draft.title} · 播放页` });
  await expect(playing).toBeVisible();
  await expect(playing.getByRole('list', { name: '歌词' }).getByRole('listitem')).toHaveText(['• • •', '第二版的第一句', '第二版的第二句']);
  await expect(playing.getByText('歌词时间按歌曲时长估算')).toBeVisible();
  // 对比试听：顶栏切换到同一作品的其他版本，歌词随版本变化；切回上一版可以点按钮或按 B。
  await playing.getByRole('button', { name: '版本 2', exact: true }).click();
  await expect(page.getByRole('menuitem')).toHaveCount(3);
  await page.getByRole('menuitem', { name: /版本 3/ }).click();
  await expect(playing.getByRole('list', { name: '歌词' }).getByRole('listitem')).toHaveText(['• • •', '第三版的歌词']);
  await expect(playing.getByRole('button', { name: '版本 3', exact: true })).toBeVisible();
  await playing.getByRole('button', { name: '切回版本 2', exact: true }).click();
  await expect(playing.getByRole('list', { name: '歌词' }).getByRole('listitem')).toHaveText(['• • •', '第二版的第一句', '第二版的第二句']);
  await page.keyboard.press('b');
  await expect(playing.getByRole('button', { name: '切回版本 2', exact: true })).toBeVisible();
  await expect(playing.getByRole('list', { name: '歌词' }).getByRole('listitem')).toHaveText(['• • •', '第三版的歌词']);
  await playing.getByRole('button', { name: '关闭播放页', exact: true }).click();
  await expect(playing).toHaveCount(0);
  await expect(rows).toHaveCount(3);

  // 复用这一版：风格和歌词写回当前作品，回到创作页。
  await rows.nth(1).getByRole('button', { name: '音频操作', exact: true }).click();
  await expect(page.getByRole('menuitem')).toHaveText(['导出音频', '复用这一版的风格和歌词']);
  await page.getByRole('menuitem', { name: '复用这一版的风格和歌词', exact: true }).click();
  await expect(page.getByLabel('风格', { exact: true })).toHaveValue('style v2');
  await expect(page.getByLabel('歌词', { exact: true })).toHaveValue(/第二版的第一句/);

  // 删除只在历史页：取消不删，确认后列表减少。
  await transport.getByRole('button', { name: '历史版本', exact: true }).click();
  await rows.nth(2).getByRole('button', { name: `删除音频${draft.title}`, exact: true }).click();
  const dialog = page.getByRole('dialog', { name: '删除这条音频？' });
  await expect(dialog).toBeVisible();
  await dialog.getByRole('button', { name: '取消', exact: true }).click();
  await expect(rows).toHaveCount(3);
  await rows.nth(2).getByRole('button', { name: `删除音频${draft.title}`, exact: true }).click();
  await page.getByRole('dialog').getByRole('button', { name: '删除音频', exact: true }).click();
  await expect(rows).toHaveCount(2);
  await expect(rows.locator('.history-meta b')).toHaveText(['版本 2', '版本 1']);
  for (let n = 2; n > 0; n--) {
    await rows.first().getByRole('button', { name: `删除音频${draft.title}`, exact: true }).click();
    await page.getByRole('dialog').getByRole('button', { name: '删除音频', exact: true }).click();
    await expect(rows).toHaveCount(n - 1);
  }
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('voice-workbench-v1')!).history);
  expect(saved).toEqual([]);
  // 版本删完后工具栏仍保留历史版本入口。
  await page.getByRole('button', { name: '返回', exact: true }).click();
  await expect(transport.getByRole('button', { name: '历史版本', exact: true })).toBeVisible();
});

test('播放页逐行校准歌词：按键标记、撤销，校准结果随版本保存并可清除', async ({ page }) => {
  const draft = { ...createMusicDraft('zh-CN'), id: 'f'.repeat(32) };
  const state = emptyState();
  state.drafts = [draft];
  state.history = [{ id: 'song', title: draft.title, fileName: 'song.wav', createdAt: '2026-09-15T12:00:00Z', duration: 6, settings: { ...draft, lyrics: '[intro]\n\n[verse]\n第一句\n第二句\n\n[chorus]\n第三句' } }];
  await page.goto('/');
  await page.evaluate(async () => {
    const rate = 8000, seconds = 6, bytes = new Uint8Array(44 + rate * seconds * 2); const view = new DataView(bytes.buffer);
    const text = (offset: number, value: string) => bytes.set(new TextEncoder().encode(value), offset);
    text(0, 'RIFF'); view.setUint32(4, bytes.length - 8, true); text(8, 'WAVEfmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, rate * seconds * 2, true);
    for (let i = 0; i < rate * seconds; i++) view.setInt16(44 + i * 2, Math.round(Math.sin(i * 220 * Math.PI * 2 / rate) * 4000), true);
    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.open('voice-workbench-audio', 1);
      request.onupgradeneeded = () => request.result.createObjectStore('audio');
      request.onerror = () => reject(request.error);
      request.onsuccess = () => { const db = request.result; const tx = db.transaction('audio', 'readwrite'); tx.objectStore('audio').put(new Blob([bytes], { type: 'audio/wav' }), 'song.wav'); tx.oncomplete = () => { db.close(); resolve(); }; };
    });
  });
  await seedState(page, state); await page.reload();
  await page.locator('.player .transport').getByRole('button', { name: '播放页', exact: true }).click();
  const playing = page.getByRole('dialog', { name: `${draft.title} · 播放页` });
  // 打开时焦点不落在关闭按钮上，避免一进来就出现焦点框和提示。
  await expect(playing.getByRole('button', { name: '关闭播放页', exact: true })).not.toBeFocused();
  await expect(playing.getByText('歌词时间按歌曲时长估算')).toBeVisible();
  await playing.getByRole('button', { name: '校准歌词', exact: true }).click();
  const sync = playing.getByRole('group', { name: '校准歌词' });
  // 开头的器乐段落从 0 秒开始，自动标好；之后每按一次 ↓ 标记下一行，退格撤销。
  await expect(sync).toContainText('1 / 4 行');
  await expect(playing.locator('.now-playing-line[data-target]')).toHaveText('第一句');
  await page.keyboard.press('ArrowDown');
  await expect(sync).toContainText('2 / 4 行');
  await page.keyboard.press('Backspace');
  await expect(sync).toContainText('1 / 4 行');
  await page.keyboard.press('ArrowDown');
  await sync.getByRole('button', { name: '下一行', exact: true }).click();
  await page.keyboard.press('ArrowDown');
  await expect(sync).toHaveCount(0);
  await expect(playing.getByText('歌词时间已校准')).toBeVisible();
  await expect(playing.getByRole('list', { name: '歌词' }).getByRole('listitem')).toHaveText(['• • •', '第一句', '第二句', '第三句']);
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('voice-workbench-v1')!).history[0].lyricTimes as number[]);
  expect(saved).toHaveLength(4);
  expect(saved[0]).toBe(0);
  expect([...saved].sort((a, b) => a - b)).toEqual(saved);
  // 清除校准后不再使用保存的时间。
  await playing.getByRole('button', { name: '清除校准', exact: true }).click();
  await expect(playing.getByText('歌词时间按歌曲时长估算')).toBeVisible();
  await expect.poll(() => page.evaluate(() => JSON.parse(localStorage.getItem('voice-workbench-v1')!).history[0].lyricTimes)).toBeUndefined();
  await playing.getByRole('button', { name: '关闭播放页', exact: true }).click();
  await expect(playing).toHaveCount(0);
});

test('播放页按音频里的人声自动定位歌词，点歌词跳到这一句实际开始唱的位置', async ({ page }) => {
  const song = syntheticSong(songLyrics, .3);
  const draft = { ...createMusicDraft('zh-CN'), id: 'e'.repeat(32) };
  const state = emptyState();
  state.drafts = [draft];
  state.history = [{ id: 'song', title: draft.title, fileName: 'song.wav', createdAt: '2026-09-15T12:00:00Z', duration: song.duration, settings: { ...draft, lyrics: songLyrics } }];
  await page.goto('/');
  await page.evaluate(async (data: string) => {
    const bytes = Uint8Array.from(atob(data), char => char.charCodeAt(0));
    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.open('voice-workbench-audio', 1);
      request.onupgradeneeded = () => request.result.createObjectStore('audio');
      request.onerror = () => reject(request.error);
      request.onsuccess = () => { const db = request.result; const tx = db.transaction('audio', 'readwrite'); tx.objectStore('audio').put(new Blob([bytes], { type: 'audio/wav' }), 'song.wav'); tx.oncomplete = () => { db.close(); resolve(); }; };
    });
  }, Buffer.from(wavBytes(song.left, song.right, song.rate)).toString('base64'));
  await seedState(page, state); await page.reload();
  await page.locator('.player .transport').getByRole('button', { name: '播放页', exact: true }).click();
  const playing = page.getByRole('dialog', { name: `${draft.title} · 播放页` });
  await expect(playing.getByText('歌词时间按人声自动定位，可再校准')).toBeVisible();
  // 点副歌第一句，播放位置跳到它实际开始唱的时间附近；点击后会继续播放，所以记录跳转那一刻的位置。
  const index = song.rows.findIndex(row => row.text === '就这样一起走到天亮');
  await playing.locator('audio').evaluate((audio: HTMLAudioElement) => audio.addEventListener('seeking', () => { audio.dataset.seekedTo = String(audio.currentTime); }));
  await playing.getByRole('list', { name: '歌词' }).getByText('就这样一起走到天亮', { exact: true }).click();
  await expect.poll(() => playing.locator('audio').evaluate((audio: HTMLAudioElement) => Number(audio.dataset.seekedTo ?? -1))).toBeGreaterThan(song.starts[index] - .6);
  expect(Number(await playing.locator('audio').getAttribute('data-seeked-to'))).toBeLessThan(song.starts[index] + .6);
});

test('新建作品默认用已下载的模型，并沿用上一次同类作品的选择', async ({ page }) => {
  const state = emptyState();
  const older = { ...createMusicDraft('zh-CN'), id: 'a'.repeat(32), modelId: 'ace-step-1.5-turbo-q8', updatedAt: '2026-10-01T00:00:00Z' };
  const recent = { ...createMusicDraft('zh-CN'), id: 'b'.repeat(32), modelId: 'yue2-3b-q4', updatedAt: '2026-10-02T00:00:00Z' };
  state.drafts = [older, recent];
  state.models = [{ id: 'ace-step-1.5-turbo-q8', path: 'ace', managed: true }, { id: 'yue2-3b-q4', path: 'yue', managed: true }];
  await seedState(page, state);
  await page.goto('/');
  await page.getByTestId('nav-new').click();
  await page.getByRole('menuitem', { name: '音乐生成', exact: true }).click();
  const model = page.getByRole('combobox', { name: '模型' });
  await expect(model).toContainText('YuE2 3B');
  // 选择器只列已下载的模型。
  await model.click();
  await expect(page.getByRole('option')).toHaveText([/^ACE-Step 1.5 Turbo\s*Q8 · 已安装/, /^YuE2 3B\s*Q4 · 已安装/, '管理模型…']);
});

test('同一模型下载了多个精度时选择器只列一项，在下方切换精度', async ({ page }) => {
  const state = emptyState();
  state.drafts = [{ ...createMusicDraft('zh-CN'), id: 'b'.repeat(32), modelId: 'yue2-3b-q4' }];
  state.models = [{ id: 'ace-step-1.5-turbo-q8', path: 'ace', managed: true }, { id: 'yue2-3b-q4', path: 'yue', managed: true }, { id: 'yue2-3b-q8', path: 'yue8', managed: true }];
  await seedState(page, state);
  await page.goto('/');
  const model = page.getByRole('combobox', { name: '模型' });
  await expect(model).toContainText('YuE2 3B · Q4');
  await model.click();
  await expect(page.getByRole('option')).toHaveText([/^ACE-Step 1.5 Turbo/, /^YuE2 3B\s*Q8 · Q4/, '管理模型…']);
  await page.keyboard.press('Escape');
  const precision = page.getByRole('radiogroup', { name: '精度' });
  await expect(precision.getByRole('radio')).toHaveText(['Q8', 'Q4']);
  await precision.getByRole('radio', { name: 'Q8' }).click();
  await expect(page.getByRole('radio', { name: 'Q8' })).toBeChecked();
});

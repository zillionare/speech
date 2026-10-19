import { test, expect, type Page } from '@playwright/test';
import { seedState } from './seed';
import { createDraft, emptyState, type Draft, type Generation } from '../src/shared/workbench';

// 资源页：汇总生成音频、配音片段、参考音频与时间轴素材，筛选、统计并批量删除未使用的项目。
async function seed(page: Page) {
  const state = emptyState();
  const story = { ...createDraft(false, 'zh-CN'), id: 's'.repeat(32), title: '第一章', kind: 'story' as const };
  const text = { ...createDraft(false, 'zh-CN'), id: 't'.repeat(32), title: '旁白练习' };
  const at = (minutes: number) => new Date(Date.UTC(2026, 9, 18, 8, minutes)).toISOString();
  const generation = (n: string, title: string, settings: Draft, minutes: number, duration: number, extra: Partial<Generation> = {}): Generation =>
    ({ id: n.repeat(32), title, fileName: n.repeat(32) + '.wav', createdAt: at(minutes), duration, settings, ...extra });
  state.history = [
    generation('a', '清晨旁白', text, 1, 4),
    generation('b', '长篇试读', text, 2, 30),
    generation('c', '被引用的版本', story, 3, 2),
    generation('d', '片段', { ...story, text: '天亮了。' }, 4, 1, { segment: { cueId: 'q', speakerId: '1', speakerName: '旁白', batchId: 'x', index: 0 } }),
  ];
  story.timeline = { tracks: [{ id: 'lane', name: '旁白', clips: [{ id: 'clip', generationId: 'c'.repeat(32), start: 0, duration: 2 }] }], assets: [{ id: 'm'.repeat(32), name: '雨声', fileName: 'import-' + 'm'.repeat(32) + '.wav', duration: 5 }] } as unknown as typeof story.timeline;
  state.drafts = [story, text];
  state.voices = [
    { id: 'v'.repeat(32), name: '我的声音', fileName: 'v'.repeat(32) + '.wav', duration: 3 },
    { id: 'w'.repeat(32), name: '闲置录音', fileName: 'w'.repeat(32) + '.wav', duration: 6 },
  ];
  state.characters = [{ id: 'k'.repeat(32), name: '林夕', settings: { ...text, voiceId: 'v'.repeat(32) }, demoText: '你好' }];
  await seedState(page, state);
}

test('资源页汇总各类资源，筛选、统计，并只能批量删除未使用的项目', async ({ page }) => {
  await seed(page);
  await page.goto('/');
  await page.getByTestId('nav-settings').click();
  await page.getByRole('tab', { name: '资源', exact: true }).click();
  const assets = page.getByTestId('assets-page');
  await expect(assets.getByText(/^共 7 项/)).toBeVisible();
  const rows = assets.locator('tbody tr');
  await expect(rows).toHaveCount(7);
  // 被时间轴引用的版本、被角色引用的参考音频、时间轴素材都不能勾选。
  await expect(rows.filter({ hasText: '被引用的版本' }).getByRole('checkbox')).toBeDisabled();
  await expect(rows.filter({ hasText: '我的声音' }).getByText('使用中：林夕')).toBeVisible();
  await expect(rows.filter({ hasText: '我的声音' }).getByRole('checkbox')).toBeDisabled();
  await expect(rows.filter({ hasText: '雨声' }).getByRole('checkbox')).toBeDisabled();
  await expect(rows.filter({ hasText: '天亮了' }).getByText('配音片段')).toBeVisible();

  // 按类型筛选：类型下拉里带各类数量。
  await assets.getByRole('combobox', { name: '类型' }).click();
  await page.getByRole('option', { name: '参考音频 · 2' }).click();
  await expect(rows).toHaveCount(2);
  await assets.getByRole('combobox', { name: '类型' }).click();
  await page.getByRole('option', { name: '全部类型' }).click();
  await expect(rows).toHaveCount(7);

  // 只看未使用的，全选后批量删除：4 项可删（两条语音版本、片段、闲置录音）。
  await assets.getByRole('combobox', { name: '使用情况' }).click();
  await page.getByRole('option', { name: '未使用' }).click();
  await expect(rows).toHaveCount(4);
  await assets.locator('thead').getByRole('checkbox').check();
  await expect(assets.getByRole('button', { name: '删除所选' })).toBeVisible();
  await assets.getByRole('button', { name: '删除所选' }).click();
  await page.getByRole('dialog', { name: '删除 4 项资源？' }).getByRole('button', { name: '删除', exact: true }).click();
  await expect(assets.getByText(/^已删除 4 项/)).toBeVisible();
  await expect(assets.getByText('没有符合条件的资源。')).toBeVisible();
  await assets.getByRole('combobox', { name: '使用情况' }).click();
  await page.getByRole('option', { name: '全部' }).click();
  await expect(rows).toHaveCount(3);
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('voice-workbench-v1') ?? '{}'));
  expect(saved.history.map((g: Generation) => g.title)).toEqual(['被引用的版本']);
  expect(saved.voices.map((v: { name: string }) => v.name)).toEqual(['我的声音']);
});

test('资源页按大小排序，并能跳到所属作品', async ({ page }) => {
  await seed(page);
  await page.goto('/');
  await page.getByTestId('nav-settings').click();
  await page.getByRole('tab', { name: '资源', exact: true }).click();
  const assets = page.getByTestId('assets-page');
  await assets.getByRole('button', { name: '大小' }).click();
  // 升序：最小的是 1 秒的配音片段；降序：最大的是 30 秒的长篇试读。
  await expect(assets.locator('tbody tr').first()).toContainText('天亮了');
  await assets.getByRole('button', { name: '大小' }).click();
  await expect(assets.locator('tbody tr').first()).toContainText('长篇试读');
  await assets.locator('tbody tr').filter({ hasText: '长篇试读' }).getByRole('button', { name: '旁白练习' }).click();
  await expect(page.getByLabel('作品名称')).toHaveValue('旁白练习');
});

import { test, expect, type Page } from '@playwright/test';
import { seedState } from './seed';
import { emptyState, type Preferences } from '../src/shared/workbench';

async function open(page: Page, preferences: Partial<Preferences>) {
  const state = emptyState();
  await seedState(page, { ...state, preferences: { ...state.preferences, ...preferences } });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
}
const saved = (page: Page) => page.evaluate(() => JSON.parse(localStorage.getItem('voice-workbench-v1')!).preferences);

test('开场画面之后接着显示功能引导，逐步高亮侧栏入口，看完记住', async ({ page }) => {
  await open(page, { introSeen: false, tourSeen: false });
  await page.getByRole('dialog', { name: 'yovoice' }).getByRole('button', { name: '开始探索' }).click();
  const tour = page.getByRole('dialog', { name: '新建作品' });
  await expect(tour).toBeVisible();
  await expect(tour.getByText('1 / 7')).toBeVisible();
  await expect(tour.getByRole('button', { name: '下一步' })).toBeFocused();
  // 高亮框框住当前目标。
  const spotlight = async () => page.locator('.guide-spotlight').boundingBox();
  const target = await page.getByTestId('nav-new').boundingBox();
  await expect.poll(async () => { const box = await spotlight(); return !!box && !!target && box.x <= target.x && box.y <= target.y && box.x + box.width >= target.x + target.width; }).toBe(true);
  await tour.getByRole('button', { name: '下一步' }).click();
  await expect(page.getByRole('dialog', { name: '语音生成' })).toBeVisible();
  // 方向键前后切换。
  await page.keyboard.press('ArrowLeft');
  await expect(page.getByRole('dialog', { name: '新建作品' })).toBeVisible();
  for (let i = 0; i < 6; i++) await page.keyboard.press('ArrowRight');
  const last = page.getByRole('dialog', { name: '随时问助手' });
  await expect(last.getByText('7 / 7')).toBeVisible();
  await expect(last.getByRole('button', { name: '跳过' })).toHaveCount(0);
  await last.getByRole('button', { name: '开始使用' }).click();
  await expect(page.locator('.guide-tour')).toHaveCount(0);
  await expect.poll(() => saved(page)).toMatchObject({ introSeen: true, tourSeen: true });
  await page.reload();
  await expect(page.getByTestId('nav-new')).toBeVisible();
  await expect(page.locator('.guide-tour')).toHaveCount(0);
});

test('功能引导可用 Esc 跳过；侧栏收起时先展开；设置里可重新查看且不改动已保存状态', async ({ page }) => {
  await open(page, { tourSeen: false });
  await expect(page.getByRole('dialog', { name: '新建作品' })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.locator('.guide-tour')).toHaveCount(0);
  await expect.poll(() => saved(page)).toMatchObject({ tourSeen: true });
  await page.getByTestId('nav-settings').click();
  await page.getByRole('button', { name: '收起侧栏' }).click();
  await expect(page.getByTestId('nav-settings')).toHaveCount(0);
  await page.getByRole('button', { name: '重新查看' }).click();
  await expect(page.getByRole('dialog', { name: '新建作品' })).toBeVisible();
  await expect(page.getByTestId('nav-new')).toBeVisible();
  await page.getByRole('dialog', { name: '新建作品' }).getByRole('button', { name: '跳过' }).click();
  await expect(page.locator('.guide-tour')).toHaveCount(0);
  await page.reload();
  await expect(page.getByTestId('nav-new')).toBeVisible();
  await expect(page.locator('.guide-tour')).toHaveCount(0);
});

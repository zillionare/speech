import type { Page } from '@playwright/test';

// 语音生成的“历史生成”在播放器工具栏右侧。
export async function openProjectHistory(page: Page) {
  await page.locator('.player .transport').getByRole('button', { name: '历史生成', exact: true }).click();
}
// “保存为角色”在右侧面板的角色选择里，保存的是面板当前的声音设置。
export async function saveAsCharacter(page: Page) {
  await page.getByRole('combobox', { name: '选择角色', exact: true }).click();
  await page.getByRole('option', { name: '保存为角色', exact: true }).click();
}

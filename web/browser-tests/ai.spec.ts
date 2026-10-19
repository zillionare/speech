import { test, expect } from '@playwright/test';

// AI 服务设置与助手入口的基础行为；对话面板的完整流程见 assistant.spec.ts。

test('AI 服务：未配置时引导到设置，左侧选服务、右侧直接编辑，切换和删除', async ({ page }) => {
  await page.goto('/');
  await page.getByTestId('assistant-toggle').click();
  const notConfigured = page.getByTestId('assistant-panel');
  await expect(notConfigured).toContainText('还没有可用的 AI 服务');
  await notConfigured.getByRole('button', { name: '设置 AI', exact: true }).click();
  await expect(page.getByRole('tab', { name: 'AI 服务', selected: true })).toBeVisible();
  // 浮动面板会盖住设置页右下角的按钮，配置时先收起。
  await page.keyboard.press('ControlOrMeta+j');
  const panel = page.getByRole('tabpanel', { name: 'AI 服务' });


  // 固定地址的服务只显示地址和协议；密钥保存后只显示掩码，不回填。
  await panel.getByRole('button', { name: 'DeepSeek', exact: true }).click();
  await expect(panel.getByText('https://api.deepseek.com', { exact: true })).toBeVisible();
  await expect(panel.getByRole('textbox', { name: '服务地址' })).toHaveCount(0);
  await expect(panel.getByRole('button', { name: '从 DeepSeek 获取 API 密钥' })).toBeVisible();
  await panel.getByLabel('API 密钥').fill('sk-preview-key');
  await panel.getByRole('textbox', { name: '当前模型' }).fill('deepseek-chat');
  await panel.getByRole('button', { name: '添加服务', exact: true }).click();
  await expect(panel.locator('.ai-nav').getByLabel('使用中')).toHaveCount(1);
  await expect(panel.getByLabel('API 密钥')).toHaveText('••••••••');
  await expect(panel.getByRole('button', { name: '保存更改', exact: true })).toBeDisabled();

  // 本地服务可改地址；新添加的服务不会抢走当前服务。
  await panel.getByRole('button', { name: 'Ollama', exact: true }).click();
  await expect(panel.getByRole('textbox', { name: '服务地址' })).toHaveValue('http://127.0.0.1:11434/v1');
  await panel.getByRole('textbox', { name: '当前模型' }).fill('qwen3');
  await panel.getByRole('button', { name: '添加服务', exact: true }).click();
  // 服务页不再显示开关和说明文字，当前服务只在左侧用状态点标出。
  await expect(panel.getByRole('switch')).toHaveCount(0);
  await expect(panel.locator('.ai-detail h2')).toHaveText('Ollama');
  await page.screenshot({ path: testInfoPath('ai-settings.png') });

  // 更换密钥：输入框为空，不回填旧密钥。
  await panel.getByRole('button', { name: 'DeepSeek', exact: true }).click();
  await panel.getByRole('button', { name: '更换密钥', exact: true }).click();
  await expect(panel.getByLabel('API 密钥')).toHaveValue('');
  const saved = await page.evaluate(() => localStorage.getItem('voice-workbench-v1')!);
  expect(saved).not.toContain('sk-preview-key');

  await panel.getByRole('button', { name: 'Ollama', exact: true }).click();
  await panel.getByRole('button', { name: '删除服务', exact: true }).click();
  await page.getByRole('dialog').getByRole('button', { name: '删除', exact: true }).click();
  await expect(panel.getByRole('button', { name: 'Ollama', exact: true })).toBeVisible();
  await expect(panel.getByText('已添加')).toBeVisible();
  await panel.getByRole('button', { name: 'DeepSeek', exact: true }).click();
  await expect(panel.locator('.ai-nav').getByLabel('使用中')).toHaveCount(1);
});

test('安装包缺少 AI 助手运行时：助手面板禁用并说明原因，不影响其他功能', async ({ page }) => {
  await page.goto('/');
  await page.evaluate(() => localStorage.setItem('yovoice-preview-no-ocho', '1'));
  await page.reload();
  await page.getByTestId('nav-settings').click();
  await page.getByRole('tab', { name: 'AI 服务' }).click();
  await page.getByRole('button', { name: 'Ollama', exact: true }).click();
  await page.getByRole('textbox', { name: '当前模型' }).fill('qwen3');
  await page.getByRole('button', { name: '添加服务', exact: true }).click();
  await page.getByTestId('nav-new').click();
  await page.getByRole('menuitem', { name: '编曲生成', exact: true }).click();
  // 编辑器里不再有单独的 AI 写作栏，AI 统一走助手；助手面板说明原因，输入不可用。
  await expect(page.locator('.ai-composer')).toHaveCount(0);
  await page.keyboard.press('ControlOrMeta+j');
  await expect(page.getByTestId('assistant-panel')).toContainText('当前安装包未包含 AI 助手运行时');
  await expect(page.getByTestId('assistant-panel').getByRole('textbox')).toHaveCount(0);
  // 其他功能不受影响：乐谱导入仍可用（导出按钮本来就因空乐谱禁用）。
  await expect(page.getByRole('button', { name: '导入', exact: true })).toBeEnabled();
});

test('OrcaRouter 一键连接：未配置时通过设置 AI 统一配置，设置页授权后切到新服务', async ({ page }) => {
  await page.goto('/');
  await page.getByTestId('assistant-toggle').click();
  const assistant = page.getByTestId('assistant-panel');
  await assistant.screenshot({ path: testInfoPath('orca-setup.png') });
  // 助手面板没有快捷连接按钮，统一从“设置 AI”去配置。
  await expect(assistant.getByRole('button', { name: '免费连接 OrcaRouter', exact: true })).toHaveCount(0);
  await assistant.getByRole('button', { name: '设置 AI', exact: true }).click();
  await page.keyboard.press('ControlOrMeta+j');

  // 打开 AI 设置，默认打开 provider 列表的第一个（OrcaRouter）。
  const panel = page.getByRole('tabpanel', { name: 'AI 服务' });
  await expect(panel.locator('.ai-detail h2')).toHaveText('OrcaRouter');
  await expect(panel.getByRole('button', { name: '免费连接 OrcaRouter', exact: true })).toBeVisible();

  // 点击“免费连接 OrcaRouter”，预览后端模拟浏览器授权完成：自动添加服务并使用免费模型。
  await panel.getByRole('button', { name: '免费连接 OrcaRouter', exact: true }).click();
  await expect(panel.getByLabel('API 密钥')).toHaveText('••••••••');
  await expect(panel.getByRole('textbox', { name: '当前模型' })).toHaveValue('orcarouter/free');
  await expect(panel.locator('.ai-nav').getByLabel('使用中')).toHaveCount(1);
  await page.screenshot({ path: testInfoPath('orca-settings.png') });

  // 打开助手，输入框随即可用。
  await page.getByTestId('assistant-toggle').click();
  await expect(assistant.getByRole('textbox')).toBeVisible();
});

test('ChatGPT 订阅（Codex）：单独分组，没有地址和密钥，显示本机 Codex CLI 状态', async ({ page }, testInfo) => {
  await page.goto('/');
  await page.getByRole('button', { name: '设置', exact: true }).click();
  await page.getByRole('tab', { name: 'AI 服务', exact: true }).click();
  const panel = page.getByRole('tabpanel', { name: 'AI 服务' });
  await expect(panel.locator('.ai-nav-head', { hasText: '订阅' })).toBeVisible();
  await panel.getByRole('button', { name: 'ChatGPT (Codex)', exact: true }).click();
  await expect(panel.getByText('Codex CLI', { exact: true })).toBeVisible();
  // 预览不能运行本机程序，显示为未安装并给出安装说明；不出现密钥和地址输入。
  await expect(panel.getByRole('status').filter({ hasText: '未找到 Codex CLI' })).toBeVisible();
  await expect(panel.getByRole('button', { name: '如何安装 Codex', exact: true })).toBeVisible();
  await expect(panel.getByLabel('API 密钥')).toHaveCount(0);
  await expect(panel.getByRole('textbox', { name: '服务地址' })).toHaveCount(0);
  await expect(panel.getByRole('button', { name: '同步模型' })).toBeEnabled();
  await page.screenshot({ path: testInfo.outputPath('codex-settings.png') });
});

// 验收截图：设置 YOVOICE_SHOTS 目录时才保存，默认不落盘。
function testInfoPath(name: string) {
  return process.env.YOVOICE_SHOTS ? `${process.env.YOVOICE_SHOTS}/${name}` : undefined;
}

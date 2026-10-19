/**
 * 界面语言注册表：新增语言只改这里、catalogs 目录和 Go 的 schema.UiLocales。
 * name 用各语言的自称，语言选择器里不随界面语言翻译。
 */
export const uiLocales = [
  { id: 'zh-CN', name: '简体中文' },
  { id: 'zh-TW', name: '繁體中文' },
  { id: 'en', name: 'English' },
  { id: 'ja', name: '日本語' },
  { id: 'ko', name: '한국어' },
  { id: 'es', name: 'Español' },
  { id: 'fr', name: 'Français' },
  { id: 'de', name: 'Deutsch' },
  { id: 'pt-BR', name: 'Português (Brasil)' },
  { id: 'ru', name: 'Русский' },
  { id: 'it', name: 'Italiano' },
  { id: 'vi', name: 'Tiếng Việt' },
] as const;

export type UiLocale = typeof uiLocales[number]['id'];

const ids = new Set<string>(uiLocales.map(locale => locale.id));

export function isUiLocale(raw: unknown): raw is UiLocale {
  return typeof raw === 'string' && ids.has(raw);
}

/** Boundary parse for preferences.uiLocale (JSON / RPC). */
/** 首次启动按系统语言选择：繁体（台港澳或 Hant）用繁體中文，其余中文用简体，葡语统一用巴西葡语，没有对应语言时用英文。 */
export function resolveInitialUiLocale(navigatorLanguage: string): UiLocale {
  const tag = navigatorLanguage.trim().toLowerCase().replaceAll('_', '-');
  const [primary] = tag.split('-');
  if (primary === 'zh') return /-(hant|tw|hk|mo)(-|$)/.test(tag) ? 'zh-TW' : 'zh-CN';
  if (primary === 'pt') return 'pt-BR';
  return isUiLocale(primary) ? primary : 'en';
}

/** 示例作品等预置内容只备有简体、繁体和英文三份，其他界面语言使用英文。 */
export type SeedLocale = 'zh-CN' | 'zh-TW' | 'en';
export const seedLocale = (locale: UiLocale): SeedLocale => locale === 'zh-CN' || locale === 'zh-TW' ? locale : 'en';

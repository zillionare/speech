import type { Catalog, MessagesByLocale } from '@astryxdesign/core/i18n';
import type { UiLocale } from '../locale';
import en from './en.json';

type Loader = () => Promise<{ default: unknown }>;

// 英文是源语言，随主包加载并作为缺词回退；其余语言切换到时再加载。
const app: Record<Exclude<UiLocale, 'en'>, Loader> = {
  'zh-CN': () => import('./zh-cn.json'),
  'zh-TW': () => import('./zh-tw.json'),
  ja: () => import('./ja.json'),
  ko: () => import('./ko.json'),
  es: () => import('./es.json'),
  fr: () => import('./fr.json'),
  de: () => import('./de.json'),
  'pt-BR': () => import('./pt-br.json'),
  ru: () => import('./ru.json'),
  it: () => import('./it.json'),
  vi: () => import('./vi.json'),
};

// Astryx 组件自带的文案（日期选择、分页等），英文已内置。
const astryx: Record<Exclude<UiLocale, 'en'>, Loader> = {
  'zh-CN': () => import('@astryxdesign/core/locales/zh-CN.json'),
  'zh-TW': () => import('@astryxdesign/core/locales/zh-TW.json'),
  ja: () => import('@astryxdesign/core/locales/ja-JP.json'),
  ko: () => import('@astryxdesign/core/locales/ko-KR.json'),
  es: () => import('@astryxdesign/core/locales/es-ES.json'),
  fr: () => import('@astryxdesign/core/locales/fr-FR.json'),
  de: () => import('@astryxdesign/core/locales/de-DE.json'),
  'pt-BR': () => import('@astryxdesign/core/locales/pt-BR.json'),
  ru: () => import('@astryxdesign/core/locales/ru-RU.json'),
  it: () => import('@astryxdesign/core/locales/it-IT.json'),
  vi: () => import('@astryxdesign/core/locales/vi-VN.json'),
};

export const enMessages = en as Catalog;

/** 某一语言交给 Provider 的完整目录：Astryx 文案 ∪ 应用英文 ∪ 应用该语言，缺词时显示英文而不是消息码。 */
export async function loadMessages(locale: UiLocale): Promise<MessagesByLocale> {
  if (locale === 'en') return { en: enMessages };
  const [own, components] = await Promise.all([app[locale](), astryx[locale]()]);
  return { [locale]: { ...(components.default as Catalog), ...enMessages, ...(own.default as Catalog) } };
}

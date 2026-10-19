import { useEffect, useState, type ReactNode } from 'react';
import { InternationalizationProvider, type MessagesByLocale } from '@astryxdesign/core/i18n';
import { loadMessages } from '../shared/i18n/catalogs';
import { isUiLocale, resolveInitialUiLocale, type UiLocale } from '../shared/i18n/locale';
import type { Preferences } from '../shared/workbench';
import { call } from '../shared/lib/client';
import { reportDesktop } from '../shared/lib/bridge';

const bootLocale = resolveInitialUiLocale(typeof navigator !== 'undefined' ? navigator.language : 'en');

export type LocaleShellProps = {
  preferencesLocale: UiLocale | null | undefined;
  children: ReactNode;
};

/**
 * Owns InternationalizationProvider locale + documentElement.lang.
 * Boot uses navigator until preferences.uiLocale arrives.
 * 目录按需加载：加载完成前保持上一种语言，首次加载完成前不渲染，避免先闪一下英文。
 */
export function LocaleShell({ preferencesLocale, children }: LocaleShellProps) {
  const wanted = isUiLocale(preferencesLocale) ? preferencesLocale : bootLocale;
  const [loaded, setLoaded] = useState<{ locale: UiLocale; messages: MessagesByLocale } | null>(null);
  useEffect(() => {
    let current = true;
    loadMessages(wanted)
      .then(messages => { if (current) setLoaded({ locale: wanted, messages }); })
      .catch(error => {
        reportDesktop('error', '', `load locale ${wanted}: ${String(error)}`);
        if (current) void loadMessages('en').then(messages => { if (current) setLoaded(prev => prev ?? { locale: 'en', messages }); });
      });
    return () => { current = false; };
  }, [wanted]);
  useEffect(() => {
    if (loaded) document.documentElement.lang = loaded.locale;
  }, [loaded]);
  if (!loaded) return null;
  return (
    <InternationalizationProvider locale={loaded.locale} messages={loaded.messages}>
      {children}
    </InternationalizationProvider>
  );
}

/** Persist navigator-derived uiLocale once when preferences omit a valid value. */
export async function ensureUiLocalePersisted(
  preferences: Preferences,
  resolved: UiLocale = bootLocale,
): Promise<void> {
  if (isUiLocale(preferences.uiLocale)) return;
  await call('preferences.save', { ...preferences, uiLocale: resolved });
}

export { bootLocale };

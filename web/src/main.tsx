import React from 'react';
import { createRoot } from 'react-dom/client';
import { Theme } from '@astryxdesign/core/theme';
import '@astryxdesign/core/reset.css';
import '@astryxdesign/core/astryx.css';
import { App } from './app/app';
import { yovoicePaperTheme } from './app/yovoice-paper';
import './app/yovoice-paper.css';
import './app/styles.css';
import { reportDesktop } from './shared/lib/bridge';
import { RootBoundary } from './shared/ui/page-boundary';
import { dismissSplash } from './shared/lib/splash';

// 未捕获的界面错误写入服务日志，排查问题时与服务端事件放在一起看；同时收起启动动画，免得挡住出错的界面。
window.addEventListener('error', event => {
  dismissSplash(true);
  reportDesktop('error', '', `${event.message} @ ${event.filename}:${event.lineno}`);
});
window.addEventListener('unhandledrejection', event => {
  dismissSplash(true);
  reportDesktop('rejection', '', String((event.reason as Error)?.stack ?? event.reason));
});
createRoot(document.getElementById('root')!).render(<React.StrictMode><Theme theme={yovoicePaperTheme} mode="light"><RootBoundary><App /></RootBoundary></Theme></React.StrictMode>);

import { useState } from 'react';
import { Button } from '@astryxdesign/core/Button';
import { HStack, VStack } from '@astryxdesign/core/Layout';
import { useTranslator } from '@astryxdesign/core/i18n';
import { Link2 } from 'lucide-react';
import { call } from '../lib/client';
import { formatCallError } from '../i18n/format';

// OrcaRouter 一键授权：本地服务生成授权页地址，在系统浏览器中登录并同意后自动添加服务、使用免费模型，
// 用户不需要复制密钥。授权完成后状态里出现服务，调用方据此切换界面。
export function OrcaConnect({ variant = 'primary', hint = false }: { variant?: 'primary' | 'secondary'; hint?: boolean }) {
  const t = useTranslator();
  const [busy, setBusy] = useState(false);
  const [waiting, setWaiting] = useState(false);
  const [error, setError] = useState('');
  async function connect() {
    setBusy(true); setError('');
    try {
      const { url } = await call<{ url: string }>('ai.connect');
      if (url) window.open(url, '_blank', 'noopener');
      setWaiting(true);
    } catch (e) { setError(formatCallError(t, e)); }
    finally { setBusy(false); }
  }
  return <VStack gap={1}>
    <HStack gap={0}><Button size="sm" variant={variant} icon={<Link2 />} label={t('@yovoice.ai.orca.connect')} isLoading={busy} onClick={() => void connect()} /></HStack>
    {waiting ? <small role="status">{t('@yovoice.ai.orca.waiting')}</small> : hint ? <small>{t('@yovoice.ai.orca.hint')}</small> : null}
    {error ? <small className="ai-composer-error" role="alert">{error}</small> : null}
  </VStack>;
}

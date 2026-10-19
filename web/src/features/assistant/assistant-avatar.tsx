import { useState } from 'react';
import { BotAvatar } from 'bot-avatars';
import { ThinkingOrb, type OrbState } from 'thinking-orbs';
import type { TurnPhase } from './assistant-model';

// Ocho 的形象：一只毛绒小幽灵（bot-avatars 的 ghost），身体取应用的强调色（暖橙），与 yovoice 的配色一致。
// 空闲时东张西望、偶尔眨眼，工作时蹦跳。
export function AssistantAvatar({ working, size }: { working: boolean; size: number }) {
  const [color] = useState(accentColor);
  return <BotAvatar className="assistant-avatar" type="ghost" color={color} state={working ? 'working' : 'default'} size={size} aria-hidden />;
}

// accentColor 把主题强调色变量解析成画布能用的颜色值（画布不认 CSS 变量）。
function accentColor(): string {
  const probe = document.createElement('span');
  probe.style.cssText = 'position:fixed;visibility:hidden;color:var(--app-accent)';
  document.body.appendChild(probe);
  const color = getComputedStyle(probe).color;
  probe.remove();
  return color || 'orange';
}

// 每个阶段一种点阵：思考是分层翻转的球，回复是起伏的彩带，查找是扫描的地球，
// 改作品是轮廓在圆、三角、方之间变形，生成音频是一圈圈滚动的声波，打开或切换是连线的星座，其余是绕轨道的粒子。
const phaseOrbs: Record<TurnPhase, OrbState> = {
  thinking: 'solving', writing: 'composing', searching: 'searching', editing: 'shaping', generating: 'listening', opening: 'connecting', working: 'working',
};

/** 运行中的状态点，与旁边带流光的阶段文字一起说明此刻在做什么。对话里用 32px，入口按钮里用 20px。 */
export function AssistantOrb({ phase, size = 20 }: { phase: TurnPhase; size?: 20 | 32 }) {
  return <ThinkingOrb className="assistant-orb" state={phaseOrbs[phase]} size={size} theme="auto" aria-hidden="true" />;
}

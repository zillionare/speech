import { useEffect, useState } from 'react';
import { VStack, HStack } from '@astryxdesign/core/Layout';

// 拖动期间只更新读数，放开时提交一次，保持一步撤销对应一次操作。
export function AudioLevel({ label, value, min, max, unit, change }: { label: string; value: number; min: number; max: number; unit: string; change: (value: number) => void }) {
  const [current, setCurrent] = useState(value);
  useEffect(() => setCurrent(value), [value]);
  return <VStack as="label" gap={1}><HStack hAlign="between" gap={4}><small>{label}</small><small>{current.toFixed(1)} {unit}</small></HStack><input type="range" aria-label={label} min={min} max={max} step={0.5} value={current} onChange={e => setCurrent(Number(e.target.value))} onPointerUp={() => change(current)} onKeyUp={() => change(current)} onBlur={() => change(current)} /></VStack>;
}

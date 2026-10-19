import { lyricRows } from '../src/features/music/lyric-timing';

// 测试用的合成歌曲，验证按音频定位歌词。
export const songLyrics = '[intro]\n\n[verse]\n夏夜的风吹过便利店\n玻璃门外霓虹在闪烁\n我数着零钱等你出现\n\n[interlude]\n\n[chorus]\n就这样一起走到天亮\n夏天不会太长\n我们慢慢地唱\n\n[outro]';

// 合成一首带前奏、间奏的歌：鼓、贝斯、宽声场铺底、略偏右的钢琴，加上居中、带共振峰和音节起伏的人声。返回每行实际开始时间。
export function syntheticSong(lyrics: string, vocal: number) {
  let seed = 7;
  const random = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2 ** 32; };
  const rate = 16000, rows = lyricRows(lyrics), starts: number[] = [], phrases: { start: number; count: number; length: number }[] = [];
  let at = 8;
  rows.forEach((row, index) => {
    if (row.instrumental) { starts.push(index ? at : 0); if (index) at += 7; return; }
    const count = [...row.text].length, length = count / 4.5 + .5;
    starts.push(at); phrases.push({ start: at, count, length }); at += length + .3 + random() * .5;
  });
  const duration = at + 6, size = Math.floor(duration * rate), left = new Float32Array(size), right = new Float32Array(size);
  const beat = 60 / 92, chords = [[220, 277, 330], [196, 247, 294], [175, 220, 262], [196, 247, 311]];
  for (let i = 0; i < size; i++) {
    const time = i / rate, beats = time / beat, phase = beats % 1, chord = chords[Math.floor(beats / 4) % 4];
    let l = .3 * Math.sin(2 * Math.PI * chord[0] / 4 * time), r = l;
    if (Math.floor(beats) % 2 === 1 && phase < .3) { const hit = (random() * 2 - 1) * Math.exp(-phase * beat * 18) * .25; l += hit; r += hit * .8; }
    const eighth = (beats * 2) % 1;
    if (eighth < .1) { const hat = (random() * 2 - 1) * Math.exp(-eighth * beat * 60) * .12; l += hat; r += hat * .4; }
    for (const note of chord) { l += .08 * Math.sin(2 * Math.PI * note * 1.003 * time); r += .08 * Math.sin(2 * Math.PI * note * .997 * time + 1); }
    const key = chord[Math.floor(beats) % 3] * 2; let piano = 0;
    for (let h = 1; h <= 4; h++) piano += Math.sin(2 * Math.PI * key * h * time) / h;
    piano *= .2 * Math.exp(-phase * beat * 4); l += piano * .6; r += piano;
    left[i] = l; right[i] = r;
  }
  for (const { start, count, length } of phrases) {
    const offset = Math.floor(start * rate), pitch = 220 + random() * 80;
    let phase = 0;
    for (let k = 0; k < Math.floor(length * rate); k++) {
      const time = k / rate, position = time / length * count;
      const envelope = Math.min(1, time / .03) * Math.min(1, (length - time) / .08) * (.55 + .45 * Math.sin(Math.PI * Math.min(1, position % 1 * 1.1)));
      const frequency = pitch * (1 + .015 * Math.sin(2 * Math.PI * 5.5 * time)) * (1 + .05 * Math.sin(Math.floor(position) * 1.7));
      phase += 2 * Math.PI * frequency / rate;
      let value = 0;
      for (let h = 1; h * frequency < 4000; h++) {
        const f = h * frequency, formant = Math.exp(-(((f - 700) / 300) ** 2)) + .6 * Math.exp(-(((f - 1200) / 400) ** 2)) + .3 * Math.exp(-(((f - 2600) / 500) ** 2)) + .2;
        value += formant * Math.sin(h * phase) / Math.sqrt(h);
      }
      value *= vocal * envelope * .4; left[offset + k] += value; right[offset + k] += value;
    }
  }
  return { rows, starts, duration, left, right, rate };
}

// 编码成 16 位立体声 WAV。
export function wavBytes(left: Float32Array, right: Float32Array, rate: number) {
  const bytes = new Uint8Array(44 + left.length * 4), view = new DataView(bytes.buffer);
  const text = (offset: number, value: string) => bytes.set(new TextEncoder().encode(value), offset);
  text(0, 'RIFF'); view.setUint32(4, bytes.length - 8, true); text(8, 'WAVEfmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 2, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 4, true); view.setUint16(32, 4, true); view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, left.length * 4, true);
  for (let i = 0; i < left.length; i++) {
    view.setInt16(44 + i * 4, Math.round(Math.max(-1, Math.min(1, left[i] * .5)) * 32767), true);
    view.setInt16(46 + i * 4, Math.round(Math.max(-1, Math.min(1, right[i] * .5)) * 32767), true);
  }
  return bytes;
}

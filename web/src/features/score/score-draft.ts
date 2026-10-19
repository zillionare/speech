import { createDraft, emptyScore, type Draft, type Score, type ScoreRole, type ScoreTrack, type UiLocale } from '../../shared/workbench';
import { seedLocale, type SeedLocale } from '../../shared/i18n/locale';

export { emptyScore };

export const soundFontId = 'musescore-general-sf2';

// 与 Go 的 schema.ScoreRoles 一致：主奏在前，铺底和琶音退后。
export const roleLevels: Record<ScoreRole, number> = { melody: -21, piano: -21, strings: -23, bass: -25, drums: -25, pad: -29, arp: -29, other: -24 };
export const scoreRoles = Object.keys(roleLevels) as ScoreRole[];
export const targetLevel = (track: ScoreTrack) => track.level || roleLevels[track.role ?? 'other'];

// 以四分音符为拍：6/8 记为 3 拍。
export const beatsPerBar = (score: Score) => score.timeSignature[0] * 4 / score.timeSignature[1];
export const barTempo = (score: Score, bar: number) => score.sections?.find(s => s.tempo && bar >= s.start && bar <= s.end)?.tempo ?? score.tempo;
export function scoreBars(score: Score) {
  const per = beatsPerBar(score);
  let bars = Math.max(0, ...(score.sections ?? []).map(s => s.end));
  for (const track of score.tracks) for (const n of track.notes) bars = Math.max(bars, Math.ceil(((n.bar - 1) * per + n.beat - 1 + n.length) / per - 1e-9));
  return bars;
}
// 逐小节累加，支持段落变速；与 Go 的 Score.Seconds 一致。
export function scoreSeconds(score: Score, bar: number, beat = 1) {
  const per = beatsPerBar(score);
  let position = (bar - 1) * per + beat - 1, seconds = 0;
  for (let b = 1; position > 0; b++) { const step = Math.min(position, per); seconds += step * 60 / barTempo(score, b); position -= step; }
  return seconds;
}
export const scoreDuration = (score: Score) => scoreSeconds(score, scoreBars(score) + 1);
export const noteCount = (score: Score) => score.tracks.reduce((sum, t) => sum + t.notes.length, 0);


// 示例乐谱里的曲名、声部和段落名，按预置内容语言选择。
const exampleNames: Record<SeedLocale, Record<'title' | 'pad' | 'piano' | 'violin' | 'viola' | 'cello' | 'arp' | 'drums' | 'key' | 'intro' | 'build' | 'outro', string>> = {
  'zh-CN': { title: '科技解读 · 垫底配乐', pad: '铺底', piano: '钢琴', violin: '小提琴', viola: '中提琴', cello: '大提琴', arp: '琶音', drums: '鼓', key: 'D 小调', intro: '前奏', build: '展开', outro: '回落' },
  'zh-TW': { title: '科技解讀 · 襯底配樂', pad: '鋪底', piano: '鋼琴', violin: '小提琴', viola: '中提琴', cello: '大提琴', arp: '琶音', drums: '鼓', key: 'D 小調', intro: '前奏', build: '展開', outro: '回落' },
  en: { title: 'Tech Explainer Underscore', pad: 'Pad', piano: 'Piano', violin: 'Violin', viola: 'Viola', cello: 'Cello', arp: 'Arp', drums: 'Drums', key: 'D minor', intro: 'Intro', build: 'Build', outro: 'Outro' },
};

// 示例：科技解读视频的垫底配乐。D 小调 100 BPM 22 小节，前奏只有铺底和钢琴，展开加入弦乐、低音、琶音和鼓，回落时钢琴放慢收在 Dm。
export function exampleScore(locale: UiLocale): Score {
  const n = exampleNames[seedLocale(locale)];
  const chords = [[50, 53, 57], [46, 50, 53], [41, 45, 48], [48, 52, 55]];
  const melody = [[[81, 2], [77, 1], [76, 1]], [[74, 1.5], [77, .5], [74, 2]], [[72, 2], [69, 1], [72, 1]], [[76, 3], [74, 1]]] as const;
  const chordAt = (bar: number) => bar >= 22 ? chords[0] : chords[(bar - 1) % 4];
  const track = (id: string, name: string, role: ScoreRole, program: number, extra: Partial<ScoreTrack> = {}): ScoreTrack => ({ id, name, role, program, notes: [], ...extra });
  const pad = track('pad', n.pad, 'pad', 89, { reverb: .5 });
  const piano = track('piano', n.piano, 'piano', 0, { reverb: .3, pan: -.15, humanize: { velocity: 8, timingMs: 6 } });
  const violin = track('violin', n.violin, 'melody', 40, { reverb: .35, pan: .2, humanize: { velocity: 6 }, dynamics: [{ start: 7, end: 9, from: .35, to: 1 }, { start: 17, end: 18, from: 1, to: .55 }] });
  const viola = track('viola', n.viola, 'strings', 41, { reverb: .35, pan: -.25, dynamics: [{ start: 7, end: 9, from: .35, to: 1 }, { start: 17, end: 18, from: 1, to: .55 }] });
  const cello = track('cello', n.cello, 'bass', 42, { reverb: .3, dynamics: [{ start: 7, end: 9, from: .35, to: 1 }] });
  const arp = track('arp', n.arp, 'arp', 81, { reverb: .4, pan: .35, humanize: { velocity: 10 } });
  const drums = track('drums', n.drums, 'drums', 0, { drums: true, humanize: { velocity: 10, timingMs: 4 } });
  const add = (t: ScoreTrack, bar: number, beat: number, pitch: number, length: number, velocity: number) => t.notes.push({ bar, beat, pitch, length, velocity });
  for (let bar = 1; bar <= 22; bar++) {
    const [root, third, fifth] = chordAt(bar);
    for (const pitch of [root, third, fifth]) add(pad, bar, 1, pitch + 12, 4, bar >= 19 ? 54 : 64);
    const arpeggio = [root, fifth, root + 12, third + 12, fifth + 12, third + 12, root + 12, fifth];
    if (bar < 19) arpeggio.forEach((pitch, i) => add(piano, bar, 1 + i / 2, pitch + 12, .5, i % 2 ? 62 : 74));
    else [root, fifth, third + 12, root + 12].forEach((pitch, i) => add(piano, bar, 1 + i, pitch + 12, bar === 22 && i === 3 ? 1 : 1.2, 66 - (bar - 19) * 4));
    if (bar >= 7 && bar <= 18) {
      let beat = 1;
      for (const [pitch, length] of melody[(bar - 1) % 4]) { add(violin, bar, beat, pitch - (bar >= 15 ? 0 : 12), length * .95, 84); beat += length; }
      add(viola, bar, 1, third + 12, 1.9, 64); add(viola, bar, 3, fifth + 12, 1.9, 60);
      add(cello, bar, 1, root - 12, 3.9, 72);
      for (let b = 1; b <= 4; b++) {
        if (b % 2) add(drums, bar, b, 36, .25, 100);
        add(drums, bar, b, 42, .25, 64); add(drums, bar, b + .5, 42, .25, 50);
        if (bar >= 13 && b % 2 === 0) add(drums, bar, b, 37, .25, 78);
      }
    }
    if (bar >= 11 && bar <= 18) for (let i = 0; i < 16; i++) add(arp, bar, 1 + i / 4, [root, third, fifth, third][i % 4] + 24, .25, i % 4 ? 52 : 64);
  }
  return {
    tempo: 100, timeSignature: [4, 4], key: n.key,
    sections: [{ name: n.intro, start: 1, end: 6 }, { name: n.build, start: 7, end: 18 }, { name: n.outro, start: 19, end: 22, tempo: 88 }],
    tracks: [pad, piano, violin, viola, cello, arp, drums],
  };
}

export function createScoreDraft(locale: UiLocale, example = true): Draft {
  return {
    ...createDraft(false, locale, soundFontId), kind: 'score',
    title: example ? exampleNames[seedLocale(locale)].title : createDraft(false, locale).title,
    score: example ? exampleScore(locale) : emptyScore(),
  };
}

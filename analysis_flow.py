import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
import matplotlib.pyplot as plt
import jaconv
import statsmodels.formula.api as smf

try:
    import japanize_matplotlib
except Exception:
    try:
        import matplotlib_fontja
    except Exception:
        print("日本語フォントを読み込めませんでした（グラフの日本語が文字化けする可能性があります）")

from google.colab import auth
auth.authenticate_user()
import gspread
from google.auth import default
creds, _ = default()
gc = gspread.authorize(creds)

SHEET_URL = 'スプレッドシートのURL'
sh = gc.open_by_url(SHEET_URL)
df = pd.DataFrame(sh.worksheet('participants').get_all_records()).replace('', np.nan)
df = df[df['実施形態'] != 'demo'].reset_index(drop=True)

RT_MAX_SEC = 120
YOIN_MIN_INDUCED = 0.05

print("Step1: データ読み込み完了")
print(f"参加者数（除外前）: {len(df)}")


def normalize_col(s):
    s = jaconv.z2h(s, kana=False, ascii=True, digit=True)
    s = s.strip()
    s = ' '.join(s.split())
    return s

def normalize_df_columns(df):
    df.columns = [normalize_col(c) for c in df.columns]
    seen = {}
    new_cols = []
    for i, col in enumerate(df.columns):
        if col not in seen:
            seen[col] = i
            new_cols.append(col)
        else:
            print(f"  ⚠ 重複列を検出・削除: '{col}'（{i+1}列目）")
            new_cols.append(f'__duplicate_{i}__')
    df.columns = new_cols
    df = df.loc[:, ~df.columns.str.startswith('__duplicate_')]
    return df

def get_trial_col(df, trial, keyword):
    matches = [c for c in df.columns if f'[試行{trial}]' in c and keyword in c]
    if len(matches) == 0:
        print(f"  ⚠ 列が見つかりません: 試行{trial} / {keyword}")
        return None
    if len(matches) > 1:
        print(f"  ⚠ 複数の列が一致しました: {matches}")
        return None
    return matches[0]

def get_col(df, keyword):
    matches = [c for c in df.columns if keyword in c]
    if len(matches) == 0:
        print(f"  ⚠ 列が見つかりません: {keyword}")
        return None
    if len(matches) > 1:
        print(f"  ⚠ 複数の列が一致しました: {matches}")
        return None
    return matches[0]


if len(df) > 0:
    df = normalize_df_columns(df)


GROUP_MAP = {
    'G1a': [('pos','pos','con'), ('mel','pos','inc'), ('pos','mel','inc'), ('mel','mel','con')],
    'G1b': [('mel','mel','con'), ('pos','mel','inc'), ('mel','pos','inc'), ('pos','pos','con')],
    'G2a': [('pos','pos','con'), ('mel','pos','inc'), ('pos','mel','inc'), ('mel','mel','con')],
    'G2b': [('mel','mel','con'), ('pos','mel','inc'), ('mel','pos','inc'), ('pos','pos','con')],
    'G3a': [('pos','mel','inc'), ('mel','mel','con'), ('pos','pos','con'), ('mel','pos','inc')],
    'G3b': [('mel','pos','inc'), ('pos','pos','con'), ('mel','mel','con'), ('pos','mel','inc')],
    'G4a': [('pos','mel','inc'), ('mel','mel','con'), ('pos','pos','con'), ('mel','pos','inc')],
    'G4b': [('mel','pos','inc'), ('pos','pos','con'), ('mel','mel','con'), ('pos','mel','inc')],
}

MATERIAL_ORDER = {
    'G1a': ['A',  "C'", "B'", 'D' ],
    'G1b': ['C',  "A'", "D'", 'B' ],
    'G2a': ['B',  "D'", "A'", 'C' ],
    'G2b': ['D',  "B'", "C'", 'A' ],
    'G3a': ["A'", 'C',  'B',  "D'"],
    'G3b': ["C'", 'A',  'D',  "B'"],
    'G4a': ["B'", 'D',  'A',  "C'"],
    'G4b': ["D'", 'B',  'C',  "A'"],
}


def resolve_aov_cols(aov):
    p_col = next((c for c in ['p-unc', 'p_unc', 'p-GG-corr', 'p_GG_corr'] if c in aov.columns), None)
    e_col = next((c for c in ['np2', 'ng2', 'n2'] if c in aov.columns), None)
    return p_col, e_col

def get_source_p(aov, source_name, p_col):
    hit = aov[aov['Source'] == source_name]
    if len(hit) == 0 or p_col is None:
        return None
    return float(hit[p_col].values[0])

def get_interaction_p(aov, p_col):
    hit = aov[aov['Source'].str.contains(r'\*', regex=True)]
    if len(hit) == 0 or p_col is None:
        return None
    return float(hit[p_col].values[0])

def verify_group_map():
    VIDEO_VAL = {'A': 'pos', 'B': 'pos', 'C': 'mel', 'D': 'mel'}

    def decode(sym):
        base, is_inc = sym[0], sym.endswith("'")
        v = VIDEO_VAL[base]
        m = ('pos' if v == 'mel' else 'mel') if is_inc else v
        return (v, m, 'inc' if is_inc else 'con')

    all_ok = True
    for g, syms in MATERIAL_ORDER.items():
        derived = [decode(s) for s in syms]
        if derived != GROUP_MAP[g]:
            print(f"  ✗ {g}: GROUP_MAPとMATERIAL_ORDERが不一致")
            print(f"      GROUP_MAP    : {GROUP_MAP[g]}")
            print(f"      MATERIAL由来 : {derived}")
            all_ok = False
    for g, conds in GROUP_MAP.items():
        n_con = sum(1 for c in conds if c[2] == 'con')
        vids = [c[0] for c in conds]
        alt = all(vids[i] != vids[i + 1] for i in range(3))
        bases = [s[0] for s in MATERIAL_ORDER[g]]
        uniq = len(set(bases)) == 4
        if not (n_con == 2 and alt and uniq):
            print(f"  ✗ {g}: 調和{n_con}回 / ポジメラ交互={alt} / 映像4種={uniq}")
            all_ok = False
    for t in range(4):
        cs = [GROUP_MAP[g][t][2] for g in GROUP_MAP]
        vs = [GROUP_MAP[g][t][0] for g in GROUP_MAP]
        if cs.count('con') != 4 or vs.count('pos') != 4:
            print(f"  ✗ 試行{t+1}: congruency con={cs.count('con')} / video pos={vs.count('pos')}")
            all_ok = False
    if all_ok:
        print("Step2: GROUP_MAP検証 → ✅ 全要件を満たしています")
    else:
        print("Step2: ⚠ GROUP_MAPに問題があります")
    return all_ok


verify_group_map()


ATTENTION_CONFIG = {
    '3を選択してください': 3,
    '6を選択してください': 6,
}

excluded_ids = []

if len(df) > 0:
    for keyword, answer in ATTENTION_CONFIG.items():
        matches = [c for c in df.columns if keyword in c]
        if matches:
            col = matches[0]
            mask = df[col] != answer
            excluded_ids.extend(df[mask].index.tolist())
        else:
            print(f"  ⚠ attention check列が見つかりません: {keyword}")

    excluded_ids = list(set(excluded_ids))
    df = df.drop(index=excluded_ids)
    print(f"\nStep3: attention check除外 {len(excluded_ids)}名 → 残り{len(df)}名")

reversed_count = 0

if len(df) > 0:
    for t in [1, 2, 3, 4]:
        col = get_trial_col(df, t, '買う気はしない')
        if col:
            df[col] = 9 - df[col]
            reversed_count += 1

    print(f"\nStep4: 逆転処理完了")
    print(f"  逆転処理した列数: {reversed_count}（想定: 4）")
    if reversed_count != 4:
        print("  ⚠ 想定と異なります。列名を確認してください。")


records = []

if len(df) > 0:
    group_col = [c for c in df.columns if 'グループID' in c]
    group_col = group_col[0] if group_col else None

    base_mood_col = get_col(df, '事前の快不快')
    base_aro_col  = get_col(df, '事前の覚醒')

    for idx, row in df.iterrows():
        group = row.get(group_col, '') if group_col else ''
        if group not in GROUP_MAP:
            continue

        base_mood = row.get(base_mood_col, np.nan) if base_mood_col else np.nan
        base_aro  = row.get(base_aro_col,  np.nan) if base_aro_col  else np.nan

        for t in [1, 2, 3, 4]:
            video_val, music_val, congruency = GROUP_MAP[group][t - 1]

            memory    = row.get(get_trial_col(df, t, '頭に残っている'), np.nan)
            mem_mood  = row.get(get_trial_col(df, t, 'この映像の雰囲気がまだ続いている'), np.nan)
            mem_world = row.get(get_trial_col(df, t, 'この映像の世界観にまだいるような'), np.nan)
            buy       = row.get(get_trial_col(df, t, '購入したいと思う'), np.nan)
            interest  = row.get(get_trial_col(df, t, '興味がある'), np.nan)
            nobuy_r   = row.get(get_trial_col(df, t, '買う気はしない'), np.nan)
            cogfit    = row.get(get_trial_col(df, t, '雰囲気が合っていた'), np.nan)
            vid_val   = row.get(get_trial_col(df, t, '映像を見てどのような気持ち'), np.nan)
            vid_aro   = row.get(get_trial_col(df, t, '映像を見てどのくらい興奮'), np.nan)
            mus_val   = row.get(get_trial_col(df, t, '音楽を聴いてどのような気持ち'), np.nan)
            mus_aro   = row.get(get_trial_col(df, t, '音楽を聴いてどのくらい興奮'), np.nan)
            liking    = row.get(get_trial_col(df, t, 'どの程度好ましく感じましたか'), np.nan)

            wtp_raw = row.get(get_trial_col(df, t, 'いくらまで払えますか'), np.nan)
            try:
                wtp_str = str(wtp_raw)
                wtp_str = jaconv.z2h(wtp_str, ascii=True, digit=True)
                wtp_str = wtp_str.replace('円', '').replace(',', '').strip()
                wtp = float(wtp_str)
            except (ValueError, TypeError):
                wtp = np.nan

            record = {
                'participant_id': idx,
                'group':          group,
                'trial':          t,
                'material':       MATERIAL_ORDER[group][t - 1],
                'video_valence':  video_val,
                'music_valence':  music_val,
                'congruency':     congruency,
                'memory':         memory,
                'mem_mood':       mem_mood,
                'mem_world':      mem_world,
                'buy':            buy,
                'interest':       interest,
                'nobuy_r':        nobuy_r,
                'wtp':            wtp,
                'cogfit':         cogfit,
                'vid_val_check':  vid_val,
                'vid_aro_check':  vid_aro,
                'mus_val_check':  mus_val,
                'mus_aro_check':  mus_aro,
                'liking':         liking,
                'base_mood':      base_mood,
                'base_arousal':   base_aro,
            }

            m1p = row.get(get_trial_col(df, t, '映像直後の快不快'), np.nan)
            m1a = row.get(get_trial_col(df, t, '映像直後の覚醒'), np.nan)
            m2p = row.get(get_trial_col(df, t, '音楽後の快不快'), np.nan)
            m2a = row.get(get_trial_col(df, t, '音楽後の覚醒'), np.nan)
            record['mood1_p'], record['mood1_a'] = m1p, m1a
            record['mood2_p'], record['mood2_a'] = m2p, m2a
            record['music_only_sec'] = row.get(get_trial_col(df, t, '音楽だけの秒数'), np.nan)
            record['hidden_music']   = row.get(get_trial_col(df, t, '音楽中のタブ離脱'), np.nan)
            record['mood1_rt']      = row.get(get_trial_col(df, t, '気分1回目の回答秒数'), np.nan)
            record['mood2_rt']      = row.get(get_trial_col(df, t, '気分2回目の回答秒数'), np.nan)
            record['yoin_item_sec'] = row.get(get_trial_col(df, t, '余韻質問の回答秒数'), np.nan)

            induced  = (m1p - base_mood) if pd.notna(m1p) and pd.notna(base_mood) else np.nan
            retained = (m2p - base_mood) if pd.notna(m2p) and pd.notna(base_mood) else np.nan
            record['induced_p']  = induced
            record['retained_p'] = retained
            if pd.notna(induced) and pd.notna(retained) and abs(induced) >= YOIN_MIN_INDUCED:
                record['yoin_rate'] = retained / induced
            else:
                record['yoin_rate'] = np.nan
            record['yoin_keep'] = (retained * np.sign(induced)) if pd.notna(induced) and pd.notna(retained) else np.nan
            record['purchase_intent'] = np.nanmean([buy, interest, nobuy_r])
            records.append(record)

long_df = pd.DataFrame(records)
print(f"\nStep5: ロング形式変換完了（{len(long_df)}行）")
if len(long_df) > 0:
    print(f"  WTP欠損値数: {long_df['wtp'].isna().sum()}件")

YOIN_ITEMS_3 = bool(len(long_df) > 0 and long_df[['memory', 'mem_world']].notna().any().any())
if len(long_df) > 0:
    print(f"\n余韻の質問：{'3問（従来の形式）' if YOIN_ITEMS_3 else '1問（補助の質問として探索的に扱う）'}")


MEMORY_ALPHA_THRESHOLD = 0.7
USE_MEMORY_SCORE = False

if len(long_df) > 0:
    alpha_data = long_df[['buy', 'interest', 'nobuy_r']].dropna()
    if len(alpha_data) > 2:
        alpha_val, _ = pg.cronbach_alpha(data=alpha_data)
        print(f"\nStep6: 購買意欲 Cronbach's α = {alpha_val:.3f}（目安 ≥ 0.7）")
        print("  項目を1つ除外したときのα:")
        for col in ['buy', 'interest', 'nobuy_r']:
            sub = alpha_data.drop(columns=[col])
            a, _ = pg.cronbach_alpha(data=sub)
            print(f"    {col}を除外: α = {a:.3f}")

    mem_alpha_data = long_df[['memory', 'mem_mood', 'mem_world']].dropna()
    if not YOIN_ITEMS_3:
        print("\n       余韻の補助質問は1問のためαは計算しない")
    if YOIN_ITEMS_3 and len(mem_alpha_data) > 2:
        mem_alpha_val, _ = pg.cronbach_alpha(data=mem_alpha_data)
        print(f"\n       余韻 Cronbach's α = {mem_alpha_val:.3f}（目安 ≥ {MEMORY_ALPHA_THRESHOLD}）")
        print("  項目を1つ除外したときのα:")
        for col in ['memory', 'mem_mood', 'mem_world']:
            sub = mem_alpha_data.drop(columns=[col])
            a, _ = pg.cronbach_alpha(data=sub)
            print(f"    {col}を除外: α = {a:.3f}")

        if mem_alpha_val >= MEMORY_ALPHA_THRESHOLD:
            USE_MEMORY_SCORE = True
            long_df['memory_score'] = long_df[['memory', 'mem_mood', 'mem_world']].mean(axis=1)
            print(f"\n  → α ≥ {MEMORY_ALPHA_THRESHOLD}：3問平均（memory_score）を使用")
        else:
            USE_MEMORY_SCORE = False
            print(f"\n  → α < {MEMORY_ALPHA_THRESHOLD}：3問を個別に分析（Bonferroni補正 α=0.017）")


if len(long_df) > 0 and 'base_mood' in long_df.columns:
    print("\nStep7-0: 事前の気分と、映像による気分の誘導の確認（Affective Slider：0〜1）")

    base = long_df.groupby('participant_id')[['base_mood', 'base_arousal']].first()
    print(f"  事前の快不快 : M={base['base_mood'].mean():.3f}, SD={base['base_mood'].std():.3f}")
    print(f"  事前の覚醒   : M={base['base_arousal'].mean():.3f}, SD={base['base_arousal'].std():.3f}")

    print("\n  映像による快不快の変化（実験前 → 映像直後）:")
    shifts = {}
    for vtype, lab in [('pos', 'ポジティブ映像'), ('mel', 'メランコリック映像')]:
        d = long_df[long_df['video_valence'] == vtype]['induced_p']
        shifts[vtype] = d.mean()
        print(f"    {lab}: {d.mean():+.3f}（SD {d.std():.3f}）")
    if pd.notna(shifts.get('pos')) and pd.notna(shifts.get('mel')):
        ok = shifts['pos'] > 0 and shifts['mel'] < 0
        print(f"    → {'✅ 意図した方向に誘導できている' if ok else '⚠ 意図した方向に誘導できていない。limitationsに記載すること'}")

    n_all = long_df['induced_p'].notna().sum()
    n_small = (long_df['induced_p'].abs() < YOIN_MIN_INDUCED).sum()
    print(f"\n  映像で気分がほぼ動かず（|変化| < {YOIN_MIN_INDUCED}）残存率を計算しなかった試行: {n_small} / {n_all}")

    if 'music_only_sec' in long_df.columns:
        mo = long_df['music_only_sec']
        print(f"  音楽だけを流した実際の秒数: M={mo.mean():.1f}, 最小={mo.min():.1f}, 最大={mo.max():.1f}")
        print(f"  音楽中にタブを離れた試行: {(long_df['hidden_music'] > 0).sum()}")

    print("\n  余韻の残存率（条件ごと）:")
    yd = long_df.groupby(['video_valence', 'music_valence', 'congruency'])['yoin_rate'].agg(['mean', 'std', 'count'])
    print(yd.to_string())
    print("  ※ 負の値：音楽の後に、映像と反対向きへ気分が振れた（音楽による上書き）")

    print("\n  覚醒（交絡の確認用）：音楽だけを聞いた後の覚醒を、音楽valence別に")
    for mtype, lab in [('pos', 'ポジティブ音楽'), ('mel', 'メランコリック音楽')]:
        a = long_df[long_df['music_valence'] == mtype]['mood2_a']
        print(f"    {lab}: M={a.mean():.3f}（SD {a.std():.3f}）")
    print("  ※ ポジ音楽とメラ音楽で覚醒が大きく違う場合、valenceとテンポ・覚醒の交絡に注意")


if len(long_df) > 0:
    print("\nStep7: 操作チェック")

    pos_vval = long_df[long_df['video_valence'] == 'pos']['vid_val_check'].mean()
    mel_vval = long_df[long_df['video_valence'] == 'mel']['vid_val_check'].mean()
    print(f"  映像valence: ポジ={pos_vval:.2f}, メラ={mel_vval:.2f}")
    print(f"  → {'✅ 意図通り（ポジ > メラ）' if pos_vval > mel_vval else '⚠ 要確認'}")

    pos_mval = long_df[long_df['music_valence'] == 'pos']['mus_val_check'].mean()
    mel_mval = long_df[long_df['music_valence'] == 'mel']['mus_val_check'].mean()
    print(f"  音楽valence: ポジ={pos_mval:.2f}, メラ={mel_mval:.2f}")
    print(f"  → {'✅ 意図通り（ポジ > メラ）' if pos_mval > mel_mval else '⚠ 要確認'}")

    per_cf = long_df.groupby(['participant_id', 'congruency'])['cogfit'].mean().unstack().dropna()
    if len(per_cf) > 1 and 'con' in per_cf and 'inc' in per_cf:
        t_stat, p_val = stats.ttest_rel(per_cf['con'], per_cf['inc'])
        print(f"\n  perceived congruency操作チェック（参加者ごとに平均して比較）:")
        print(f"  congruent M={per_cf['con'].mean():.2f} vs incongruent M={per_cf['inc'].mean():.2f}")
        print(f"  対応ありt検定: t({len(per_cf) - 1})={t_stat:.3f}, p={p_val:.3f} {'*' if p_val < 0.05 else 'n.s.'}")

    print("\n  素材ごとの「合っていた」（調和のはずなのに低い組がないか）:")
    print(long_df.groupby('material')['cogfit'].agg(['mean', 'count']).round(2).to_string())

    liking_purchase_corr = long_df[['liking', 'purchase_intent']].dropna().corr()
    r = liking_purchase_corr.loc['liking', 'purchase_intent']
    print(f"\n  映像への好意度 × 購買意欲の相関: r={r:.3f}")
    if abs(r) >= 0.3:
        print(f"  ⚠ 相関が高め（r≥.3）→ limitationsに交絡の可能性を明記すること")
    else:
        print(f"  → 相関は低い（r<.3）→ 交絡の影響は小さいと考えられる")

    pos_varo = long_df[long_df['video_valence'] == 'pos']['vid_aro_check'].mean()
    mel_varo = long_df[long_df['video_valence'] == 'mel']['vid_aro_check'].mean()
    print(f"\n  映像arousal（探索的）: ポジ={pos_varo:.2f}, メラ={mel_varo:.2f}")


if len(long_df) > 0:
    print("\nStep7-1: 余韻と購買意欲の関係（探索的検討）")

    mem_col = 'memory_score' if USE_MEMORY_SCORE else None

    for ycol, ylab in [('yoin_rate', '余韻（気分の残存率）'), ('mem_mood', '余韻の補助質問')]:
        for pcol, plab in [('purchase_intent', '購買意欲'), ('wtp', 'WTP')]:
            if ycol in long_df.columns:
                pair = long_df[[ycol, pcol]].dropna()
                if len(pair) > 2:
                    print(f"  {ylab} × {plab}: r={pair.corr().loc[ycol, pcol]:.3f}（n={len(pair)}試行）")

    if not YOIN_ITEMS_3:
        pass
    elif mem_col:
        pair = long_df[[mem_col, 'purchase_intent']].dropna()
        if len(pair) > 2:
            r_mp = pair.corr().loc[mem_col, 'purchase_intent']
            print(f"  余韻（3問平均） × 購買意欲: r={r_mp:.3f}（n={len(pair)}試行）")
    else:
        for col, lab in [('memory', '記憶'), ('mem_mood', '感情持続'), ('mem_world', '没入持続')]:
            pair = long_df[[col, 'purchase_intent']].dropna()
            if len(pair) > 2:
                r_mp = pair.corr().loc[col, 'purchase_intent']
                print(f"  余韻（{lab}） × 購買意欲: r={r_mp:.3f}（n={len(pair)}試行）")

    print("  ※ 記述的な相関であり、媒介関係を示すものではない")
    print("  ※ 同一参加者の複数試行を含むため、独立性の仮定は満たしていない")

    if 'base_mood' in long_df.columns:
        print("\n  事前の感情と各従属変数の相関（探索的）:")
        dv_list = [('purchase_intent', '購買意欲'), ('wtp', 'WTP')]
        for base_col, base_lab in [('base_mood', '事前の快不快'), ('base_arousal', '事前の覚醒')]:
            for dv_col, dv_lab in dv_list:
                pair_b = long_df[[base_col, dv_col]].dropna()
                if len(pair_b) > 2 and pair_b[base_col].nunique() > 1:
                    r_b = pair_b.corr().loc[base_col, dv_col]
                    print(f"    {base_lab} × {dv_lab}: r={r_b:.3f}")


if len(df) > 0:
    print("\nStep7-2: demand認知チェック")

    demand_col = [c for c in df.columns if '実験の目的' in c]
    if demand_col:
        demand_col = demand_col[0]
        print(f"  回答一覧（目視確認が必要）:")
        for idx, ans in df[demand_col].items():
            print(f"    参加者{idx}: {ans}")

        purpose_keywords = ['音楽', '購買', '買う', '購入', '値段', '金額', 'BGM', '支払']
        purpose_aware_ids = []
        for idx, ans in df[demand_col].items():
            ans_str = str(ans)
            hit = sum(1 for k in purpose_keywords if k in ans_str)
            if hit >= 2:
                purpose_aware_ids.append(idx)
        print(f"\n  【判定1】実験目的を推測していた可能性: {len(purpose_aware_ids)}名")
        if purpose_aware_ids:
            print(f"    参加者ID: {purpose_aware_ids}")

        manip_keywords = ['合ってない', '合っていない', '意図的', 'わざと', '違和感',
                          '不自然', 'ミスマッチ', '変えて', '組み合わせ', '合わない']
        manip_aware_ids = []
        for idx, ans in df[demand_col].items():
            ans_str = str(ans)
            hit = sum(1 for k in manip_keywords if k in ans_str)
            if hit >= 1:
                manip_aware_ids.append(idx)
        print(f"\n  【判定2】条件操作に気づいていた可能性: {len(manip_aware_ids)}名")
        if manip_aware_ids:
            print(f"    参加者ID: {manip_aware_ids}")

        print("\n  ※ 自動判定は参考値。必ず回答内容を目視確認すること")
    else:
        print("  ⚠ demand認知チェック列が見つかりません")


if len(long_df) > 0:
    print("\nStep8: 記述統計")
    dvs_desc = ['yoin_rate', 'purchase_intent', 'wtp', 'cogfit']
    if USE_MEMORY_SCORE:
        dvs_desc.insert(1, 'memory_score')

    desc = long_df.groupby(['video_valence', 'music_valence'])[dvs_desc].agg(
        ['mean', 'std', 'min', 'max', 'count']
    )
    print(desc.to_string())

    print(f"\n  WTP最大値: {long_df['wtp'].max():.0f}円")
    print(f"  WTP最小値: {long_df['wtp'].min():.0f}円")
    print(f"  WTP中央値: {long_df['wtp'].median():.0f}円")

    if 'base_mood' in long_df.columns:
        base_desc = long_df.groupby('participant_id')[['base_mood', 'base_arousal']].first()
        print("\n  事前の感情測定（参加者単位）:")
        print(base_desc.describe().loc[['mean', 'std', 'min', 'max']].to_string())

    lang_col = [c for c in df.columns if c == '言語']
    if lang_col:
        print(f"\n  回答言語: {df[lang_col[0]].value_counts().to_dict()}")

    print("\n  グループごとの参加者数:")
    gcount = long_df.groupby('group')['participant_id'].nunique()
    for g in ['G1a','G1b','G2a','G2b','G3a','G3b','G4a','G4b']:
        n = gcount.get(g, 0)
        print(f"    {g}: {n}名")
    if gcount.nunique() > 1:
        print("  ⚠ グループ間で人数が不均等です")


if len(long_df) > 0:
    plot_cols = ['yoin_rate', 'purchase_intent', 'wtp', 'cogfit']
    plot_labels_hist = ['余韻（気分の残存率）', '購買意欲', 'WTP（円）', 'perceived congruency']
    if USE_MEMORY_SCORE:
        plot_cols.insert(2, 'memory_score')
        plot_labels_hist.insert(2, '余韻持続（3問平均）')

    fig, axes = plt.subplots(1, len(plot_cols), figsize=(4 * len(plot_cols), 4))
    if len(plot_cols) == 1:
        axes = [axes]
    for ax, dv, label in zip(axes, plot_cols, plot_labels_hist):
        ax.hist(long_df[dv].dropna(), bins=15, edgecolor='black')
        ax.set_title(label)
        ax.set_xlabel('スコア')
        ax.set_ylabel('頻度')
    plt.suptitle('各従属変数の分布（ANOVA前確認）')
    plt.tight_layout()
    plt.savefig('distributions.png', dpi=150)
    plt.show()
    print("\nStep9: 分布確認グラフ保存 → distributions.png")


def art_rm_2x2(df_, dv, subj='participant_id', a='video_valence', b='music_valence'):
    d = df_[[subj, a, b, dv]].dropna().copy()
    gm = d[dv].mean()
    cell = d.groupby([a, b])[dv].transform('mean')
    ma = d.groupby(a)[dv].transform('mean')
    mb = d.groupby(b)[dv].transform('mean')
    resid = d[dv] - cell
    est = {'A': ma - gm, 'B': mb - gm, 'AB': cell - ma - mb + gm}
    out = {}
    def rows_of(aov_):
        src = aov_['Source'].astype(str)
        return {'A': aov_[src == a], 'B': aov_[src == b], 'AB': aov_[src.str.contains(r'\*')]}
    for key, e in est.items():
        d['_aligned'] = resid + e
        chk_rows = rows_of(pg.rm_anova(data=d, dv='_aligned', within=[a, b], subject=subj, detailed=True))
        chk = max((float(chk_rows[k]['F'].iloc[0]) for k in chk_rows if k != key and len(chk_rows[k])), default=np.nan)
        d['_rank'] = d['_aligned'].rank(method='average')
        aov_ = pg.rm_anova(data=d, dv='_rank', within=[a, b], subject=subj, detailed=True)
        pcol_, _ = resolve_aov_cols(aov_)
        tgt = rows_of(aov_)[key]
        out[key] = (float(tgt['F'].iloc[0]), float(tgt[pcol_].iloc[0]), chk)
    return out

def wilcoxon_diff(diff):
    d = pd.Series(diff).dropna()
    d = d[d != 0]
    if len(d) < 3:
        return None
    w = stats.wilcoxon(d)
    ranks = d.abs().rank(method='average')
    r_plus, r_minus = ranks[d > 0].sum(), ranks[d < 0].sum()
    return dict(W=float(w.statistic), p=float(w.pvalue), r=float((r_plus - r_minus) / (r_plus + r_minus)), n=len(d))


print("\nStep10: 分析開始")

if not YOIN_ITEMS_3:
    dvs = {
        'yoin_rate':       '余韻（気分の残存率）※仮説1',
        'purchase_intent': '購買意欲 ※仮説2',
        'wtp':             'WTP ※仮説2',
        'mem_mood':        '余韻の補助質問（映像の雰囲気がまだ続いている）※探索的',
    }
elif USE_MEMORY_SCORE:
    dvs = {
        'yoin_rate':       '余韻（気分の残存率）※仮説1',
        'purchase_intent': '購買意欲',
        'memory_score':    '余韻持続（3問平均）',
        'wtp':             'WTP',
    }
else:
    dvs = {
        'yoin_rate':       '余韻（気分の残存率）※仮説1',
        'purchase_intent': '購買意欲',
        'memory':          '余韻（記憶）',
        'mem_mood':        '余韻（感情持続）',
        'mem_world':       '余韻（没入持続）',
        'wtp':             'WTP',
    }
    print("⚠ 余韻を3問個別に分析します。有意水準はBonferroni補正でα=0.017を適用。")

if len(long_df) > 0:
    for dv, label in dvs.items():
        print(f"\n{'='*40}")
        print(f"【{label}】")

        data = long_df[['participant_id', 'video_valence', 'music_valence', 'congruency', dv]].dropna()

        if dv == 'wtp':
            stat, p_norm = stats.shapiro(data[dv].dropna())
            print(f"  Shapiro-Wilk検定（記録用）: W={stat:.3f}, p={p_norm:.3f}")
            print(f"  → WTPは一律で対数変換（log(WTP+1)）を適用")
            data = data.copy()
            data[dv] = np.log1p(data[dv])

        n_cells = data.groupby('participant_id').size()
        complete_ids = n_cells[n_cells == 4].index
        n_dropped = data['participant_id'].nunique() - len(complete_ids)
        if n_dropped > 0:
            print(f"  ※ 4条件が揃わない参加者{n_dropped}名をこの分析から除外（欠損のため）")
        data = data[data['participant_id'].isin(complete_ids)]

        if data['participant_id'].nunique() < 3:
            print("  ⚠ 有効な参加者が3名未満のためANOVAを実行できません")
            continue

        try:
            aov = pg.rm_anova(
                data=data,
                dv=dv,
                within=['video_valence', 'music_valence'],
                subject='participant_id',
                detailed=True
            )
            p_col, e_col = resolve_aov_cols(aov)
            show_cols = ['Source', 'F'] + [c for c in [p_col, e_col] if c]
            print(aov[show_cols].to_string(index=False))
            if e_col == 'ng2':
                print("  ※ 効果量は一般化イータ二乗（ng2）")

            ALPHA = 0.05 / 3 if (YOIN_ITEMS_3 and not USE_MEMORY_SCORE and dv in ('memory', 'mem_mood', 'mem_world')) else 0.05

            interaction_p = get_interaction_p(aov, p_col)
            per = data.groupby(['participant_id', 'congruency'])[dv].mean().unstack()
            d_ci = per['con'] - per['inc']
            n_s = len(d_ci)
            m_d, sd_d = d_ci.mean(), d_ci.std(ddof=1)
            t_c = m_d / (sd_d / np.sqrt(n_s))
            p_c = 2 * stats.t.sf(abs(t_c), n_s - 1)
            dz = m_d / sd_d
            half = stats.t.ppf(0.975, n_s - 1) * sd_d / np.sqrt(n_s)
            print(f"\n  【Step1】仮説の検証：交互作用（＝調和 vs 不調和）")
            if dv == 'wtp':
                print("    ※ WTPの平均・差は対数変換後（log(WTP+1)）の値")
            print(f"    調和 M={per['con'].mean():.2f} vs 不調和 M={per['inc'].mean():.2f}"
                  f"（差 {m_d:+.2f}、95%CI [{m_d - half:+.2f}, {m_d + half:+.2f}]）")
            print(f"    t({n_s - 1})={t_c:.2f}, p={p_c:.3f}, dz={dz:.2f}"
                  f"　（ANOVAの交互作用 p={interaction_p:.3f}：同じ検定）")
            if p_c < ALPHA and m_d > 0:
                print(f"    → 仮説の方向（調和 > 不調和）に有意（α={ALPHA:.3f}）")
            elif p_c < ALPHA and m_d < 0:
                print(f"    → 仮説と逆の方向（不調和 > 調和）に有意（α={ALPHA:.3f}）")
            else:
                print(f"    → 有意差なし（α={ALPHA:.3f}）。探索的研究なので効果量と信頼区間で大きさを読む")

            print(f"\n  【頑健性の確認】")
            if len(d_ci) >= 3:
                sw = stats.shapiro(d_ci)
                skew = stats.skew(d_ci)
                print(f"    差の得点の正規性：Shapiro-Wilk W={sw.statistic:.3f}, p={sw.pvalue:.3f}／歪度={skew:+.2f}"
                      f"{'（正規分布から外れている可能性）' if sw.pvalue < .05 else ''}")
            wres = wilcoxon_diff(d_ci)
            if wres:
                print(f"    ウィルコクソンの符号順位検定：W={wres['W']:.1f}, p={wres['p']:.3f}, r={wres['r']:+.2f}（n={wres['n']}）")
            try:
                art = art_rm_2x2(data, dv)
                lab = {'A': '映像valenceの主効果', 'B': '音楽valenceの主効果', 'AB': '交互作用（＝調和 vs 不調和）'}
                for k in ['AB', 'B', 'A']:
                    F_, p_, chk = art[k]
                    print(f"    ART {lab[k]}：F={F_:.2f}, p={p_:.3f}{'  *' if p_ < ALPHA else ''}")
                worst = max(v[2] for v in art.values())
                if not (worst < 1e-6):
                    print(f"    ⚠ ARTの整列の確認：整列させていない効果のFが0になっていない（最大{worst:.4f}）")
            except Exception as e:
                art = None
                print(f"    ART を実行できませんでした: {e}")
            verdicts = {'主な分析（t検定）': p_c < ALPHA}
            if wres: verdicts['ウィルコクソン'] = wres['p'] < ALPHA
            if art: verdicts['ART'] = art['AB'][1] < ALPHA
            if len(set(verdicts.values())) == 1:
                print(f"    → 3つの方法で結論は一致（{'有意' if p_c < ALPHA else '有意でない'}）")
            else:
                detail = '、'.join(f"{k}：{'有意' if v else '有意でない'}" for k, v in verdicts.items())
                print(f"    → 方法によって結論が分かれた（{detail}）。報告では全部を示し、慎重に解釈する")

            music_p = get_source_p(aov, 'music_valence', p_col)
            if music_p is not None:
                pos_m = data[data['music_valence'] == 'pos'][dv].mean()
                mel_m = data[data['music_valence'] == 'mel'][dv].mean()
                sig = '*' if music_p < ALPHA else 'n.s.'
                print(f"\n  【Step2】音楽valenceの主効果（副次的）: p={music_p:.3f} {sig}")
                print(f"    ポジ音楽 M={pos_m:.2f} vs メラ音楽 M={mel_m:.2f}")

            video_p = get_source_p(aov, 'video_valence', p_col)
            if video_p is not None:
                pos_v = data[data['video_valence'] == 'pos'][dv].mean()
                mel_v = data[data['video_valence'] == 'mel'][dv].mean()
                sig = '*' if video_p < ALPHA else 'n.s.'
                print(f"\n  【Step3】映像valenceの主効果（副次的）: p={video_p:.3f} {sig}")
                print(f"    ポジ映像 M={pos_v:.2f} vs メラ映像 M={mel_v:.2f}")

            alpha_s = ALPHA / 2
            note = '' if p_c < ALPHA else '（交互作用が有意でないため参考）'
            print(f"\n  【Step4】単純主効果：映像ごとに 調和 vs 不調和（α={alpha_s:.3f}）{note}")
            for vtype, label_v, con_mus in [('pos', 'ポジティブ映像', 'pos'), ('mel', 'メランコリック映像', 'mel')]:
                sub = data[data['video_valence'] == vtype]
                con_s = sub[sub['music_valence'] == con_mus].set_index('participant_id')[dv]
                inc_s = sub[sub['music_valence'] != con_mus].set_index('participant_id')[dv]
                common = con_s.index.intersection(inc_s.index)
                if len(common) > 1:
                    dd = con_s[common] - inc_s[common]
                    t_s, p_s = stats.ttest_rel(con_s[common], inc_s[common])
                    dz_s = dd.mean() / dd.std(ddof=1)
                    sig = '*' if p_s < alpha_s else 'n.s.'
                    print(f"    {label_v}: 調和 M={con_s[common].mean():.2f} vs 不調和 M={inc_s[common].mean():.2f}"
                          f"（差 {dd.mean():+.2f}）, p={p_s:.3f} {sig}, dz={dz_s:.2f}")

        except Exception as e:
            print(f"  ANOVA実行エラー（データ不足の可能性）: {e}")


def congruency_contrast(df_, col):
    per = df_.groupby(['participant_id', 'congruency'])[col].mean().unstack().dropna()
    if len(per) < 3 or 'con' not in per or 'inc' not in per:
        return None
    d = per['con'] - per['inc']
    n = len(d); m = d.mean(); sd = d.std(ddof=1)
    t_ = m / (sd / np.sqrt(n)); p_ = 2 * stats.t.sf(abs(t_), n - 1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return dict(n=n, con=per['con'].mean(), inc=per['inc'].mean(), m=m, lo=m - half, hi=m + half, t=t_, p=p_, dz=m / sd)

if len(long_df) > 0 and 'mood2_rt' in long_df.columns:
    print("\nStep10-2: 回答時間（探索的）")
    rt = long_df.copy()
    for col in ['mood1_rt', 'mood2_rt', 'yoin_item_sec']:
        bad = (rt[col] <= 0) | (rt[col] > RT_MAX_SEC)
        if bad.sum() > 0:
            print(f"  {col}: {RT_MAX_SEC}秒を超える等の {int(bad.sum())} 試行を除外")
        rt.loc[bad, col] = np.nan
        rt[col + '_log'] = np.log(rt[col])

    print("  ※ 値は対数（秒）。差がマイナス＝不調和の方が時間がかかった")
    for col, lab in [('mood1_rt_log', '気分1回目（音楽の前・比較の基準）'),
                     ('mood2_rt_log', '気分2回目（音楽の後）'),
                     ('yoin_item_sec_log', '余韻の質問')]:
        r = congruency_contrast(rt, col)
        if r:
            sig = '*' if r['p'] < 0.05 else 'n.s.'
            print(f"    {lab}: 調和 {np.exp(r['con']):.1f}秒 vs 不調和 {np.exp(r['inc']):.1f}秒"
                  f"（対数の差 {r['m']:+.3f}、95%CI [{r['lo']:+.3f}, {r['hi']:+.3f}]）, p={r['p']:.3f} {sig}, dz={r['dz']:.2f}")
    print("  → 気分1回目に差がなく、音楽の後だけ不調和が長ければ「不調和で迷いが生じた」解釈と整合")

    try:
        w = rt[['participant_id', 'wtp', 'mood2_rt_log', 'congruency']].dropna().copy()
        w['wtp_log'] = np.log1p(w['wtp'])
        w['congruency_num'] = (w['congruency'] == 'con').astype(int)
        w['rt_c'] = w['mood2_rt_log'] - w.groupby('participant_id')['mood2_rt_log'].transform('mean')
        res = smf.mixedlm("wtp_log ~ rt_c + congruency_num", data=w, groups=w['participant_id']).fit(reml=True)
        b, p_ = res.params.get('rt_c', np.nan), res.pvalues.get('rt_c', np.nan)
        print(f"\n  気分2回目の回答時間 → WTP（混合効果モデル、参加者内で中心化）: 係数={b:+.3f}, p={p_:.3f}")
        print("  → 係数がマイナス：その人の中で迷った試行ほどWTPが低い")
    except Exception as e:
        print(f"  回答時間とWTPのモデルを実行できませんでした: {e}")


if len(long_df) > 0 and 'mood2_a' in long_df.columns:
    print("\nStep10-3: 覚醒の交絡（探索的）")
    try:
        a = long_df[['participant_id', 'wtp', 'congruency', 'mood2_a', 'base_arousal']].dropna().copy()
        a['wtp_log'] = np.log1p(a['wtp'])
        a['congruency_num'] = (a['congruency'] == 'con').astype(int)
        a['aro_c'] = a['mood2_a'] - a.groupby('participant_id')['mood2_a'].transform('mean')
        a['base_c'] = a['base_arousal'] - a['base_arousal'].mean()
        res = smf.mixedlm("wtp_log ~ congruency_num + aro_c + base_c", data=a, groups=a['participant_id']).fit(reml=True)
        for term, lab in [('congruency_num', '調和（1）vs 不調和（0）'),
                          ('aro_c', '音楽後の覚醒（参加者内で中心化）'),
                          ('base_c', '実験前の覚醒（疲れの代わり）')]:
            print(f"    {lab}: 係数={res.params.get(term, np.nan):+.3f}, p={res.pvalues.get(term, np.nan):.3f}")
        print("  → 覚醒を入れても調和の係数が残れば、調和の効果は覚醒だけでは説明できない")
    except Exception as e:
        print(f"  覚醒のモデルを実行できませんでした: {e}")


if len(long_df) > 0:
    plot_dvs = ['yoin_rate', 'purchase_intent', 'wtp']
    plot_labels = ['余韻（気分の残存率）', '購買意欲', 'WTP（円）']
    if USE_MEMORY_SCORE:
        plot_dvs.insert(2, 'memory_score')
        plot_labels.insert(2, '余韻持続（3問平均）')

    x_pos = {'pos': 0, 'mel': 1}
    fig, axes = plt.subplots(1, len(plot_dvs), figsize=(5 * len(plot_dvs), 5))
    if len(plot_dvs) == 1:
        axes = [axes]

    for ax, dv, label in zip(axes, plot_dvs, plot_labels):
        summary = long_df.groupby(['video_valence', 'music_valence'])[dv].agg(['mean', 'sem']).reset_index()
        for vtype, color in zip(['pos', 'mel'], ['#E07B54', '#5B8DB8']):
            sub = summary[summary['video_valence'] == vtype].copy()
            sub['x'] = sub['music_valence'].map(x_pos)
            sub = sub.sort_values('x')
            ax.errorbar(
                sub['x'], sub['mean'], yerr=sub['sem'],
                label='ポジティブ映像' if vtype == 'pos' else 'メランコリック映像',
                color=color, marker='o', linewidth=2, capsize=5
            )
        ax.set_title(label)
        ax.set_xlabel('音楽valence')
        ax.set_ylabel('平均スコア（変換前）')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['ポジティブ', 'メランコリック'])
        ax.set_xlim(-0.5, 1.5)
        ax.legend()

    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    plt.show()
    print("\nStep11: 結果グラフ保存 → results.png")


print("\nStep12: WTPアンカリング検証（補助分析）")

if len(long_df) > 0 and 'wtp' in long_df.columns:
    print("\n--- 主分析：混合効果モデル（試行順序 × congruency）---")
    print("  ※ 探索的補助分析。アンカリング・疲労効果・学習効果の可能性を区別できない")

    wtp_data = long_df[['participant_id', 'trial', 'congruency', 'wtp']].dropna()
    wtp_data = wtp_data.copy()
    wtp_data['wtp_log'] = np.log1p(wtp_data['wtp'])
    wtp_data['congruency_num'] = (wtp_data['congruency'] == 'con').astype(int)

    try:
        model = smf.mixedlm(
            "wtp_log ~ trial * congruency_num",
            data=wtp_data,
            groups=wtp_data["participant_id"]
        )
        result = model.fit(reml=True)
        print(result.summary())

        interaction_p = result.pvalues.get('trial:congruency_num', None)
        if interaction_p is not None:
            print(f"\n  交互作用（trial × congruency）: p={interaction_p:.3f}")
            if interaction_p < 0.05:
                print("  → 試行順序によってcongruencyのWTPへの効果が変化している")
                print("  ※ 疲労効果・学習効果等の可能性も排除できない")
            else:
                print("  → 順序効果を示唆する明確なパターンは見られない")

    except Exception as e:
        print(f"  混合効果モデル実行エラー: {e}")

    print("\n--- 補助分析：試行1との差分（WTPの変化を可視化）---")

    trial1_wtp = long_df[long_df['trial'] == 1][['participant_id', 'congruency', 'wtp']].copy()
    trial1_wtp = trial1_wtp.rename(columns={'wtp': 'wtp_trial1'})

    wtp_diff = long_df[long_df['trial'] > 1][['participant_id', 'trial', 'congruency', 'wtp']].copy()
    wtp_diff = wtp_diff.merge(
        trial1_wtp[['participant_id', 'wtp_trial1']],
        on='participant_id', how='left'
    )
    wtp_diff['diff_from_trial1'] = wtp_diff['wtp'] - wtp_diff['wtp_trial1']

    diff_summary = wtp_diff.groupby(['trial', 'congruency'])['diff_from_trial1'].agg(
        ['mean', 'std']
    ).reset_index()
    print(diff_summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    wtp_by_trial = long_df.groupby(['trial', 'congruency'])['wtp'].mean().reset_index()
    for cond, color, label in zip(['con', 'inc'], ['#E07B54', '#5B8DB8'], ['congruent', 'incongruent']):
        sub = wtp_by_trial[wtp_by_trial['congruency'] == cond]
        ax1.plot(sub['trial'], sub['wtp'], marker='o', color=color, label=label, linewidth=2)
    ax1.set_title('試行ごとのWTP平均')
    ax1.set_xlabel('試行')
    ax1.set_ylabel('WTP（円）')
    ax1.set_xticks([1, 2, 3, 4])
    ax1.legend()

    ax2 = axes[1]
    for cond, color, label in zip(['con', 'inc'], ['#E07B54', '#5B8DB8'], ['congruent', 'incongruent']):
        sub = diff_summary[diff_summary['congruency'] == cond]
        ax2.errorbar(
            sub['trial'], sub['mean'], yerr=sub['std'],
            marker='o', color=color, label=label, linewidth=2, capsize=5
        )
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.set_title('試行1からの差分（アンカリング確認）')
    ax2.set_xlabel('試行')
    ax2.set_ylabel('WTP差分（円）')
    ax2.set_xticks([2, 3, 4])
    ax2.legend()

    plt.tight_layout()
    plt.savefig('anchoring_check.png', dpi=150)
    plt.show()
    print("\nグラフ保存: anchoring_check.png")

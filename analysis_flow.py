# ============================================
# 特別演習I 分析フロー v11
# 映像valence(2) × 音楽valence(2) Two-way repeated measures ANOVA
# ============================================

# ============================================
# 事前インストール
# ============================================
# !pip install pingouin japanize-matplotlib jaconv statsmodels -q

import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
import matplotlib.pyplot as plt
import japanize_matplotlib
import jaconv
import statsmodels.formula.api as smf

# ============================================
# Step1: データ読み込み
# ============================================
# GoogleスプレッドシートからCSVとしてエクスポートして読み込む
# df = pd.read_csv('responses.csv')
df = pd.DataFrame()  # 実際はCSV読み込みに差し替え

print("Step1: データ読み込み完了")
print(f"参加者数（除外前）: {len(df)}")

# ============================================
# 列名の正規化関数
# ============================================
def normalize_col(s):
    """列名の表記ゆれを吸収する"""
    s = jaconv.z2h(s, kana=False, ascii=True, digit=True)
    s = s.strip()
    s = ' '.join(s.split())
    return s

def normalize_df_columns(df):
    """全列名を正規化し、重複列を処理する"""
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
    df = df.loc[:, ~df.columns.str.startswith('__duplicate__')]
    return df

def get_trial_col(df, trial, keyword):
    """試行番号とキーワードから列名を取得"""
    matches = [c for c in df.columns if f'[試行{trial}]' in c and keyword in c]
    if len(matches) == 0:
        print(f"  ⚠ 列が見つかりません: 試行{trial} / {keyword}")
        return None
    if len(matches) > 1:
        print(f"  ⚠ 複数の列が一致しました: {matches}")
        return None
    return matches[0]

if len(df) > 0:
    df = normalize_df_columns(df)
    print("\n正規化後の列名一覧:")
    for c in df.columns:
        print(f"  {c}")

# ============================================
# Step2: グループIDから条件ラベル付与の定義
# ============================================
# 実験デザイン：映像valence × 音楽valenceの2×2
# congruent  = 映像と音楽のvalenceが一致
# incongruent = 映像と音楽のvalenceが不一致
GROUP_MAP = {
    #         試行1                  試行2                  試行3                  試行4
    # video_valence: pos=ポジティブ, mel=メランコリック
    # music_valence: pos=ポジティブ, mel=メランコリック
    # congruency: con=congruent, inc=incongruent
    'G1a': [('pos','pos','con'), ('mel','mel','con'), ('pos','mel','inc'), ('mel','pos','inc')],
    'G1b': [('mel','mel','con'), ('pos','pos','con'), ('mel','pos','inc'), ('pos','mel','inc')],
    'G2a': [('pos','pos','con'), ('mel','mel','con'), ('pos','mel','inc'), ('mel','pos','inc')],
    'G2b': [('mel','mel','con'), ('pos','pos','con'), ('mel','pos','inc'), ('pos','mel','inc')],
    'G3a': [('pos','mel','inc'), ('mel','pos','inc'), ('pos','pos','con'), ('mel','mel','con')],
    'G3b': [('mel','pos','inc'), ('pos','mel','inc'), ('mel','mel','con'), ('pos','pos','con')],
    'G4a': [('pos','mel','inc'), ('mel','pos','inc'), ('pos','pos','con'), ('mel','mel','con')],
    'G4b': [('mel','pos','inc'), ('pos','mel','inc'), ('mel','mel','con'), ('pos','pos','con')],
}

print("\nStep2: グループIDと条件ラベルの定義完了")

# ============================================
# Step3: attention checkで除外（2問）
# ============================================
# 試行2：正解3、試行4：正解6
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

# ============================================
# Step4: 逆転項目の処理（除外後に実行）
# ============================================
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

# ============================================
# Step5: ロング形式に変換 + 条件ラベル付与
# ============================================
records = []

if len(df) > 0:
    group_col = [c for c in df.columns if 'グループID' in c]
    group_col = group_col[0] if group_col else None

    for idx, row in df.iterrows():
        group = row.get(group_col, '') if group_col else ''
        if group not in GROUP_MAP:
            continue
        for t in [1, 2, 3, 4]:
            video_val, music_val, congruency = GROUP_MAP[group][t-1]

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

            # WTPの文字列・欠損値処理
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
                'video_valence':  video_val,   # pos / mel
                'music_valence':  music_val,   # pos / mel
                'congruency':     congruency,  # con / inc
                'memory':         memory,
                'mem_mood':       mem_mood,
                'mem_world':      mem_world,
                'buy':            buy,
                'interest':       interest,
                'nobuy_r':        nobuy_r,
                'wtp':            wtp,
                'cogfit':         cogfit,
                'vid_val_check':  vid_val,     # 操作チェック用
                'vid_aro_check':  vid_aro,
                'mus_val_check':  mus_val,
                'mus_aro_check':  mus_aro,
                'liking':         liking,      # 映像への好意度（交絡変数）
            }
            record['purchase_intent'] = np.nanmean([buy, interest, nobuy_r])
            records.append(record)

long_df = pd.DataFrame(records)
print(f"\nStep5: ロング形式変換完了（{len(long_df)}行）")
print(f"  WTP欠損値数: {long_df['wtp'].isna().sum()}件")

# ============================================
# Step6: Cronbach's α
# ============================================
MEMORY_ALPHA_THRESHOLD = 0.7  # 目安。内容的妥当性も考慮して判断すること
USE_MEMORY_SCORE = False

if len(long_df) > 0:
    # 購買意欲のα
    alpha_data = long_df[['buy', 'interest', 'nobuy_r']].dropna()
    if len(alpha_data) > 2:
        alpha_val, _ = pg.cronbach_alpha(data=alpha_data)
        print(f"\nStep6: 購買意欲 Cronbach's α = {alpha_val:.3f}（目安 ≥ 0.7）")
        print("  項目を1つ除外したときのα:")
        for col in ['buy', 'interest', 'nobuy_r']:
            sub = alpha_data.drop(columns=[col])
            a, _ = pg.cronbach_alpha(data=sub)
            print(f"    {col}を除外: α = {a:.3f}")

    # 余韻のα
    mem_alpha_data = long_df[['memory', 'mem_mood', 'mem_world']].dropna()
    if len(mem_alpha_data) > 2:
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
            print("  ※ 内容的妥当性も踏まえて最終判断すること")

# ============================================
# Step7: 操作チェック（valence/arousal）
# ============================================
if len(long_df) > 0:
    print("\nStep7: 操作チェック")

    # 映像valenceの確認：ポジ映像 > メラ映像になっているか
    pos_vval = long_df[long_df['video_valence']=='pos']['vid_val_check'].mean()
    mel_vval = long_df[long_df['video_valence']=='mel']['vid_val_check'].mean()
    print(f"  映像valence: ポジ={pos_vval:.2f}, メラ={mel_vval:.2f}")
    print(f"  → {'✅ 意図通り（ポジ > メラ）' if pos_vval > mel_vval else '⚠ 要確認'}")

    # 音楽valenceの確認：ポジ音楽 > メラ音楽になっているか
    pos_mval = long_df[long_df['music_valence']=='pos']['mus_val_check'].mean()
    mel_mval = long_df[long_df['music_valence']=='mel']['mus_val_check'].mean()
    print(f"  音楽valence: ポジ={pos_mval:.2f}, メラ={mel_mval:.2f}")
    print(f"  → {'✅ 意図通り（ポジ > メラ）' if pos_mval > mel_mval else '⚠ 要確認'}")

    # cognitive fitの操作チェック：congruent > incongruent
    con_cf = long_df[long_df['congruency']=='con']['cogfit'].mean()
    inc_cf = long_df[long_df['congruency']=='inc']['cogfit'].mean()
    con_data = long_df[long_df['congruency']=='con'].set_index('participant_id')['cogfit']
    inc_data = long_df[long_df['congruency']=='inc'].set_index('participant_id')['cogfit']
    common = con_data.index.intersection(inc_data.index)
    if len(common) > 1:
        t_stat, p_val = stats.ttest_rel(con_data[common], inc_data[common])
        print(f"\n  cognitive fit操作チェック:")
        print(f"  congruent M={con_cf:.2f} vs incongruent M={inc_cf:.2f}")
        print(f"  対応ありt検定: t={t_stat:.3f}, p={p_val:.3f} {'*' if p_val < 0.05 else 'n.s.'}")

    # 映像への好意度と購買意欲の相関（交絡確認）
    liking_purchase_corr = long_df[['liking', 'purchase_intent']].dropna().corr()
    r = liking_purchase_corr.loc['liking', 'purchase_intent']
    print(f"\n  映像への好意度 × 購買意欲の相関: r={r:.3f}")
    if abs(r) >= 0.3:
        print(f"  ⚠ 相関が高め（r≥.3）→ limitationsに交絡の可能性を明記すること")
    else:
        print(f"  → 相関は低い（r<.3）→ 交絡の影響は小さいと考えられる")

    # arousalは探索的に把握
    pos_varo = long_df[long_df['video_valence']=='pos']['vid_aro_check'].mean()
    mel_varo = long_df[long_df['video_valence']=='mel']['vid_aro_check'].mean()
    print(f"\n  映像arousal（探索的）: ポジ={pos_varo:.2f}, メラ={mel_varo:.2f}")

# ============================================
# Step7-2: demand認知チェック
# ============================================
# 参加者が実験の意図に気づいていたかを確認する
# 「音楽が購買意欲に影響するか」を正確に推測した参加者を特定する
if len(df) > 0:
    print("\nStep7-2: demand認知チェック")

    demand_col = [c for c in df.columns if '実験の目的' in c]
    if demand_col:
        demand_col = demand_col[0]
        print(f"  回答一覧（目視確認が必要）:")
        for idx, ans in df[demand_col].items():
            print(f"    参加者{idx}: {ans}")

        # キーワードによる自動判定（参考値）
        # 「音楽」「購買」「買う」等が含まれる回答を抽出
        keywords = ['音楽', '購買', '買う', '購入', '値段', '金額', 'BGM']
        aware_ids = []
        for idx, ans in df[demand_col].items():
            ans_str = str(ans)
            hit = sum(1 for k in keywords if k in ans_str)
            if hit >= 2:  # 2つ以上のキーワードが含まれる場合
                aware_ids.append(idx)

        print(f"\n  実験意図に気づいた可能性のある参加者: {len(aware_ids)}名")
        if aware_ids:
            print(f"    参加者ID: {aware_ids}")
        print("  ※ 自動判定は参考値。必ず回答内容を目視確認すること")
        print("  ※ 除外するかどうかは目視確認後に判断する")
    else:
        print("  ⚠ demand認知チェック列が見つかりません")

# ============================================
# Step8: 記述統計
# ============================================
if len(long_df) > 0:
    print("\nStep8: 記述統計")
    dvs_desc = ['purchase_intent', 'wtp', 'cogfit']
    if USE_MEMORY_SCORE:
        dvs_desc.insert(1, 'memory_score')

    desc = long_df.groupby(['video_valence', 'music_valence'])[dvs_desc].agg(
        ['mean', 'std', 'min', 'max', 'count']
    )
    print(desc.to_string())

    print(f"\n  WTP最大値: {long_df['wtp'].max():.0f}円")
    print(f"  WTP最小値: {long_df['wtp'].min():.0f}円")
    print(f"  WTP中央値: {long_df['wtp'].median():.0f}円")

# ============================================
# Step9: 分布の確認（ANOVA前）
# ============================================
if len(long_df) > 0:
    plot_cols = ['purchase_intent', 'wtp', 'cogfit']
    plot_labels_hist = ['購買意欲', 'WTP（円）', 'cognitive fit']
    if USE_MEMORY_SCORE:
        plot_cols.insert(1, 'memory_score')
        plot_labels_hist.insert(1, '余韻持続（3問平均）')

    fig, axes = plt.subplots(1, len(plot_cols), figsize=(4*len(plot_cols), 4))
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

# ============================================
# Step10: 分析
# ============================================
print("\nStep10: 分析開始")

if USE_MEMORY_SCORE:
    dvs = {
        'purchase_intent': '購買意欲',
        'memory_score':    '余韻持続（3問平均）',
        'wtp':             'WTP',
    }
else:
    dvs = {
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

        try:
            aov = pg.rm_anova(
                data=data,
                dv=dv,
                within=['video_valence', 'music_valence'],
                subject='participant_id',
                detailed=True
            )
            print(aov[['Source', 'F', 'p-unc', 'np2']].to_string(index=False))

            # Step1：音楽条件（music_valence）の主効果（仮説の直接検証）
            music_p = aov[aov['Source'] == 'music_valence']['p-unc'].values
            if len(music_p) > 0:
                pos_m = data[data['music_valence'] == 'pos'][dv].mean()
                mel_m = data[data['music_valence'] == 'mel'][dv].mean()
                sig = '*' if music_p[0] < 0.05 else 'n.s.'
                print(f"\n  【Step1】音楽valenceの主効果: p={music_p[0]:.3f} {sig}")
                print(f"    ポジ音楽 M={pos_m:.2f} vs メラ音楽 M={mel_m:.2f}")

            # Step2：映像valenceの主効果（副次的知見）
            video_p = aov[aov['Source'] == 'video_valence']['p-unc'].values
            if len(video_p) > 0:
                pos_v = data[data['video_valence'] == 'pos'][dv].mean()
                mel_v = data[data['video_valence'] == 'mel'][dv].mean()
                sig = '*' if video_p[0] < 0.05 else 'n.s.'
                print(f"\n  【Step2】映像valenceの主効果: p={video_p[0]:.3f} {sig}")
                print(f"    ポジ映像 M={pos_v:.2f} vs メラ映像 M={mel_v:.2f}")

            # Step3：交互作用（映像×音楽のvalenceの組み合わせ効果）
            interaction_p = aov[aov['Source'].str.contains('video_valence.*music_valence|music_valence.*video_valence')]['p-unc'].values
            if len(interaction_p) > 0:
                sig = '*' if interaction_p[0] < 0.05 else 'n.s.'
                print(f"\n  【Step3】交互作用（映像valence × 音楽valence）: p={interaction_p[0]:.3f} {sig}")

                # Step4：交互作用が有意なら単純主効果（Bonferroni補正：α=0.025）
                if interaction_p[0] < 0.05:
                    print(f"  → congruent条件とincongruent条件で効果が異なる可能性")
                    print(f"  【Step4】単純主効果（Bonferroni補正 α=0.025）")
                    for vtype in ['pos', 'mel']:
                        label_v = 'ポジティブ映像' if vtype == 'pos' else 'メランコリック映像'
                        sub = data[data['video_valence'] == vtype]
                        pos_mus = sub[sub['music_valence'] == 'pos'].set_index('participant_id')[dv]
                        mel_mus = sub[sub['music_valence'] == 'mel'].set_index('participant_id')[dv]
                        common = pos_mus.index.intersection(mel_mus.index)
                        if len(common) > 1:
                            t_stat, p_val = stats.ttest_rel(pos_mus[common], mel_mus[common])
                            sig = '*' if p_val < 0.025 else 'n.s.'
                            print(f"    {label_v}: ポジ音楽M={pos_mus.mean():.2f} vs メラ音楽M={mel_mus.mean():.2f}, p={p_val:.3f} {sig}")

        except Exception as e:
            print(f"  ANOVA実行エラー（データ不足の可能性）: {e}")

# ============================================
# Step11: 可視化（変換前スコアで表示）
# ============================================
if len(long_df) > 0:
    plot_dvs = ['purchase_intent', 'wtp']
    plot_labels = ['購買意欲', 'WTP（円）']
    if USE_MEMORY_SCORE:
        plot_dvs.insert(1, 'memory_score')
        plot_labels.insert(1, '余韻持続（3問平均）')

    fig, axes = plt.subplots(1, len(plot_dvs), figsize=(5*len(plot_dvs), 5))
    if len(plot_dvs) == 1:
        axes = [axes]

    for ax, dv, label in zip(axes, plot_dvs, plot_labels):
        summary = long_df.groupby(['video_valence', 'music_valence'])[dv].agg(['mean', 'sem']).reset_index()
        for vtype, color in zip(['pos', 'mel'], ['#E07B54', '#5B8DB8']):
            sub = summary[summary['video_valence'] == vtype]
            ax.errorbar(
                sub['music_valence'], sub['mean'], yerr=sub['sem'],
                label='ポジティブ映像' if vtype == 'pos' else 'メランコリック映像',
                color=color, marker='o', linewidth=2, capsize=5
            )
        ax.set_title(label)
        ax.set_xlabel('音楽valence')
        ax.set_ylabel('平均スコア（変換前）')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['ポジティブ', 'メランコリック'])
        ax.legend()

    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    plt.show()
    print("\nStep11: 結果グラフ保存 → results.png")

# ============================================
# Step12: WTPアンカリング検証（補助分析）
# ============================================
print("\nStep12: WTPアンカリング検証（補助分析）")

if len(long_df) > 0 and 'wtp' in long_df.columns:

    # --- 主分析：混合効果モデル（試行順序 × congruency）---
    # rm ANOVAを使わない理由：
    # 各参加者の各試行にはcongruent/incongruentどちらかのデータしかない（セルが不完全）
    # 混合効果モデルは不完全なセル構造でも対応できる
    # 注意：WTPの収束はアンカリング以外に疲労効果・学習効果の可能性も排除できない
    # 探索的補助分析として位置づける
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
                print("  → アンカリングを含む順序効果の可能性と整合的")
                print("  ※ 疲労効果・学習効果等の可能性も排除できない")
            else:
                print("  → 順序効果を示唆する明確なパターンは見られない")

    except Exception as e:
        print(f"  混合効果モデル実行エラー: {e}")

    # --- 補助分析：試行1との差分の可視化 ---
    print("\n--- 補助分析：試行1との差分（WTPの変化を可視化）---")
    print("  ※ 統計的検定ではなく記述・可視化が目的")

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
    print("差分が0に近づいていく → アンカリングなどの順序効果と整合的なパターン")
    print("差分が明確に収束しない → そのようなパターンは明確ではない")
    print("※ いずれもアンカリングの有無を直接証明するものではない")
    print("※ 主分析（混合効果モデル）と合わせて解釈すること")

# ============================================
# 特別演習I 分析フロー v2
# 映像種類(2) × 音楽条件(2) Two-way repeated measures ANOVA
# ============================================

# ============================================
# 事前インストール
# ============================================
# !pip install pingouin japanize-matplotlib jaconv -q

import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
import matplotlib.pyplot as plt
import japanize_matplotlib
import jaconv

# ============================================
# Step1: データ読み込み
# ============================================
# GoogleスプレッドシートからCSVとしてエクスポートして読み込む
# df = pd.read_csv('responses.csv')
# 以下はテスト用ダミーデータで動作確認
df = pd.DataFrame()  # 実際はCSV読み込みに差し替え

print("Step1: データ読み込み完了")
print(f"参加者数（除外前）: {len(df)}")

# ============================================
# 列名の正規化関数
# 全角→半角、前後スペース削除、空白統一
# ============================================
def normalize_col(s):
    """列名の表記ゆれを吸収する"""
    s = jaconv.z2h(s, kana=False, ascii=True, digit=True)  # 全角英数記号→半角
    s = s.strip()           # 前後スペース削除
    s = ' '.join(s.split()) # 連続スペースを1つに統一
    return s

def normalize_df_columns(df):
    """DataFrameの全列名を正規化"""
    df.columns = [normalize_col(c) for c in df.columns]
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

# 列名を正規化
if len(df) > 0:
    df = normalize_df_columns(df)
    print("\n正規化後の列名一覧:")
    for c in df.columns:
        print(f"  {c}")

# ============================================
# Step2: グループIDから条件ラベル付与
# ============================================

# 8グループ×4試行のマッピングテーブル
# 条件の組み合わせ（pos/mel × con/dis）はG1aとG2aで同じ
# 異なるのは刺激の提示順（どの映像を試行1に使うか）
# → カウンターバランスの目的は「順序効果の統制」であり、条件の割り当てが同じことは意図通り

GROUP_MAP = {
    #         試行1              試行2              試行3              試行4
    # 映像：A=ポジ①滝, B=ポジ②夕日, C=メラ①ベンチ, D=メラ②紅葉（'はそれぞれ不調和）
    'G1a': [('pos','con'), ('mel','dis'), ('pos','dis'), ('mel','con')],  # A  C' B' D
    'G1b': [('mel','con'), ('pos','dis'), ('mel','dis'), ('pos','con')],  # C  A' D' B
    'G2a': [('pos','con'), ('mel','dis'), ('pos','dis'), ('mel','con')],  # B  D' A' C  ※G1aと条件組み合わせ同一・提示順が異なる
    'G2b': [('mel','con'), ('pos','dis'), ('mel','dis'), ('pos','con')],  # D  B' C' A
    'G3a': [('pos','dis'), ('mel','con'), ('pos','con'), ('mel','dis')],  # A' C  B  D'
    'G3b': [('mel','dis'), ('pos','con'), ('mel','con'), ('pos','dis')],  # C' A  D  B'
    'G4a': [('pos','dis'), ('mel','con'), ('pos','con'), ('mel','dis')],  # B' D  A  C' ※G3aと条件組み合わせ同一・提示順が異なる
    'G4b': [('mel','dis'), ('pos','con'), ('mel','con'), ('pos','dis')],  # D' B  C  A'
}

print("\nStep2: グループIDと条件ラベルの定義完了")

# ============================================
# Step3: attention checkで除外（逆転処理より先に実行）
# ============================================
ATTENTION_ANSWER = 4  # UIで設定した正解値

before = len(df)
excluded_ids = []

if len(df) > 0:
    attention_col = 'この質問では4を選択してください'
    # 列名正規化後に検索
    matches = [c for c in df.columns if '4を選択' in c]
    if matches:
        attention_col = matches[0]
        mask = df[attention_col] != ATTENTION_ANSWER
        excluded_ids = df[mask].index.tolist()
        df = df.drop(index=excluded_ids)
        print(f"\nStep3: attention check除外 {len(excluded_ids)}名 → 残り{len(df)}名")
    else:
        print("\nStep3: attention check列が見つかりません。スキップします。")

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
            video_type, music_cond = GROUP_MAP[group][t-1]

            buy      = row.get(get_trial_col(df, t, '購入したいと思う'), np.nan)
            interest = row.get(get_trial_col(df, t, '興味がある'), np.nan)
            nobuy_r  = row.get(get_trial_col(df, t, '買う気はしない'), np.nan)  # 逆転済み
            memory   = row.get(get_trial_col(df, t, '頭に残っている'), np.nan)
            cogfit   = row.get(get_trial_col(df, t, '雰囲気が合っていた'), np.nan)

            # WTPの文字列・欠損値処理
            wtp_raw = row.get(get_trial_col(df, t, 'いくらまで払えますか'), np.nan)
            try:
                wtp_str = str(wtp_raw)
                wtp_str = jaconv.z2h(wtp_str, ascii=True, digit=True)  # 全角数字→半角
                wtp_str = wtp_str.replace('円', '').replace(',', '').strip()
                wtp = float(wtp_str)
            except (ValueError, TypeError):
                wtp = np.nan  # 「なし」「空欄」等はNaNに

            record = {
                'participant_id': idx,
                'group':          group,
                'trial':          t,
                'video_type':     video_type,
                'music_cond':     music_cond,
                'memory':         memory,
                'buy':            buy,
                'interest':       interest,
                'nobuy_r':        nobuy_r,
                'wtp':            wtp,
                'cogfit':         cogfit,
            }
            record['purchase_intent'] = np.nanmean([buy, interest, nobuy_r])
            records.append(record)

long_df = pd.DataFrame(records)
print(f"\nStep5: ロング形式変換完了（{len(long_df)}行）")
print(f"  WTP欠損値数: {long_df['wtp'].isna().sum()}件")

# ============================================
# Step6: 購買意欲スケールの平均算出 + Cronbach's α
# ============================================
if len(long_df) > 0:
    alpha_data = long_df[['buy', 'interest', 'nobuy_r']].dropna()
    if len(alpha_data) > 2:
        alpha_val, alpha_ci = pg.cronbach_alpha(data=alpha_data)
        print(f"\nStep6: Cronbach's α = {alpha_val:.3f}（目標 ≥ 0.7）")

        # 項目を1つ抜いたときのαの変化
        print("  項目を1つ除外したときのα:")
        for col in ['buy', 'interest', 'nobuy_r']:
            sub = alpha_data.drop(columns=[col])
            a, _ = pg.cronbach_alpha(data=sub)
            print(f"    {col}を除外: α = {a:.3f}")

# ============================================
# Step7: 分布の確認（ANOVA前）
# ============================================
if len(long_df) > 0:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, dv, label in zip(axes,
                              ['purchase_intent', 'memory', 'wtp', 'cogfit'],
                              ['購買意欲', '余韻持続', 'WTP（円）', 'cognitive fit']):
        ax.hist(long_df[dv].dropna(), bins=15, edgecolor='black')
        ax.set_title(label)
        ax.set_xlabel('スコア')
        ax.set_ylabel('頻度')
    plt.suptitle('各従属変数の分布（ANOVA前確認）')
    plt.tight_layout()
    plt.savefig('distributions.png', dpi=150)
    plt.show()
    print("\nStep7: 分布確認グラフ保存 → distributions.png")

# ============================================
# Step8: 分析
# ============================================
print("\nStep8: 分析開始")

dvs = {
    'purchase_intent': '購買意欲',
    'memory':          '余韻持続',
    'wtp':             'WTP',
    'cogfit':          'cognitive fit（操作チェック）',
}

if len(long_df) > 0:
    for dv, label in dvs.items():
        print(f"\n{'='*40}")
        print(f"【{label}】")

        data = long_df[['participant_id', 'video_type', 'music_cond', dv]].dropna()

        if dv == 'cogfit':
            # 操作チェック：対応ありt検定（同じ参加者が両条件を経験）
            con = data[data['music_cond'] == 'con'].set_index('participant_id')[dv]
            dis = data[data['music_cond'] == 'dis'].set_index('participant_id')[dv]
            common = con.index.intersection(dis.index)
            t_stat, p_val = stats.ttest_rel(con[common], dis[common])
            print(f"  調和条件 M={con.mean():.2f}, 不調和条件 M={dis.mean():.2f}")
            print(f"  対応ありt検定: t={t_stat:.3f}, p={p_val:.3f} {'*' if p_val < 0.05 else 'n.s.'}")

        else:
            if dv == 'wtp':
                # WTPは結果によらず対数変換（右歪みが確実なため）
                stat, p_norm = stats.shapiro(data[dv].dropna())
                print(f"  Shapiro-Wilk検定（記録用）: W={stat:.3f}, p={p_norm:.3f}")
                print(f"  → WTPは一律で対数変換（log(WTP+1)）を適用")
                data = data.copy()
                data[dv] = np.log1p(data[dv])
                print(f"  ※ ANOVAは対数変換後、グラフは変換前のスコアで表示")

            # Two-way repeated measures ANOVA
            try:
                aov = pg.rm_anova(
                    data=data,
                    dv=dv,
                    within=['video_type', 'music_cond'],
                    subject='participant_id',
                    detailed=True
                )
                print(aov[['Source', 'F', 'p-unc', 'np2']].to_string(index=False))

                # 交互作用が有意なら単純主効果（Bonferroni補正：α=0.05/2=0.025）
                interaction_p = aov[aov['Source'].str.contains('video_type.*music_cond|music_cond.*video_type')]['p-unc'].values
                if len(interaction_p) > 0 and interaction_p[0] < 0.05:
                    print(f"\n  → 交互作用有意（p={interaction_p[0]:.3f}）")
                    print(f"  → 単純主効果の比較（Bonferroni補正 α=0.025）")
                    for vtype in ['pos', 'mel']:
                        label_v = 'ポジティブ' if vtype == 'pos' else 'メランコリック'
                        sub = data[data['video_type'] == vtype]
                        con = sub[sub['music_cond'] == 'con'].set_index('participant_id')[dv]
                        dis = sub[sub['music_cond'] == 'dis'].set_index('participant_id')[dv]
                        common = con.index.intersection(dis.index)
                        if len(common) > 1:
                            t_stat, p_val = stats.ttest_rel(con[common], dis[common])
                            sig = '*' if p_val < 0.025 else 'n.s.'  # Bonferroni補正
                            print(f"    {label_v}: 調和M={con.mean():.2f} vs 不調和M={dis.mean():.2f}, p={p_val:.3f} {sig}")

            except Exception as e:
                print(f"  ANOVA実行エラー（データ不足の可能性）: {e}")

# ============================================
# Step9: 可視化（変換前スコアで表示）
# ============================================
if len(long_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_dvs    = ['purchase_intent', 'memory', 'wtp']
    plot_labels = ['購買意欲', '余韻持続', 'WTP（円）']

    for ax, dv, label in zip(axes, plot_dvs, plot_labels):
        summary = long_df.groupby(['video_type', 'music_cond'])[dv].agg(['mean', 'sem']).reset_index()
        for vtype, color in zip(['pos', 'mel'], ['#E07B54', '#5B8DB8']):
            sub = summary[summary['video_type'] == vtype]
            ax.errorbar(
                sub['music_cond'], sub['mean'], yerr=sub['sem'],
                label='ポジティブ' if vtype == 'pos' else 'メランコリック',
                color=color, marker='o', linewidth=2, capsize=5
            )
        ax.set_title(label)
        ax.set_xlabel('音楽条件')
        ax.set_ylabel('平均スコア（変換前）')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['調和', '不調和'])
        ax.legend()

    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    plt.show()
    print("\nStep9: 結果グラフ保存 → results.png")

# ============================================
# 特別演習I 分析フロー v8
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
    """
    全列名を正規化し、重複列を処理する。
    Googleフォームのバグで同じ列名が複数出ることがある。
    その場合は最初の列だけを残して重複を削除する。
    """
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
GROUP_MAP = {
    #         試行1              試行2              試行3              試行4
    # 映像：A=ポジ①滝, B=ポジ②夕日, C=メラ①ベンチ, D=メラ②紅葉（'はそれぞれ不調和）
    # 条件の組み合わせ（pos/mel × con/dis）はG1aとG2aで同じ
    # 異なるのは刺激の提示順→順序効果の統制が目的
    'G1a': [('pos','con'), ('mel','dis'), ('pos','dis'), ('mel','con')],  # A  C' B' D
    'G1b': [('mel','con'), ('pos','dis'), ('mel','dis'), ('pos','con')],  # C  A' D' B
    'G2a': [('pos','con'), ('mel','dis'), ('pos','dis'), ('mel','con')],  # B  D' A' C
    'G2b': [('mel','con'), ('pos','dis'), ('mel','dis'), ('pos','con')],  # D  B' C' A
    'G3a': [('pos','dis'), ('mel','con'), ('pos','con'), ('mel','dis')],  # A' C  B  D'
    'G3b': [('mel','dis'), ('pos','con'), ('mel','con'), ('pos','dis')],  # C' A  D  B'
    'G4a': [('pos','dis'), ('mel','con'), ('pos','con'), ('mel','dis')],  # B' D  A  C'
    'G4b': [('mel','dis'), ('pos','con'), ('mel','con'), ('pos','dis')],  # D' B  C  A'
}

print("\nStep2: グループIDと条件ラベルの定義完了")

# ============================================
# Step3: attention checkで除外（逆転処理より先に実行）
# ============================================
ATTENTION_ANSWER = 4

excluded_ids = []

if len(df) > 0:
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

            buy       = row.get(get_trial_col(df, t, '購入したいと思う'), np.nan)
            interest  = row.get(get_trial_col(df, t, '興味がある'), np.nan)
            nobuy_r   = row.get(get_trial_col(df, t, '買う気はしない'), np.nan)
            memory    = row.get(get_trial_col(df, t, '頭に残っている'), np.nan)
            mem_mood  = row.get(get_trial_col(df, t, 'この映像の雰囲気がまだ続いている'), np.nan)
            mem_world = row.get(get_trial_col(df, t, 'この映像の世界観にまだいるような'), np.nan)
            cogfit    = row.get(get_trial_col(df, t, '雰囲気が合っていた'), np.nan)
            vid_val   = row.get(get_trial_col(df, t, '映像を見てどのような気持ち'), np.nan)
            vid_aro   = row.get(get_trial_col(df, t, '映像を見てどのくらい興奮'), np.nan)
            mus_val   = row.get(get_trial_col(df, t, '音楽を聴いてどのような気持ち'), np.nan)
            mus_aro   = row.get(get_trial_col(df, t, '音楽を聴いてどのくらい興奮'), np.nan)

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
                'video_type':     video_type,
                'music_cond':     music_cond,
                'memory':         memory,
                'mem_mood':       mem_mood,
                'mem_world':      mem_world,
                'buy':            buy,
                'interest':       interest,
                'nobuy_r':        nobuy_r,
                'wtp':            wtp,
                'cogfit':         cogfit,
                'video_valence':  vid_val,
                'video_arousal':  vid_aro,
                'music_valence':  mus_val,
                'music_arousal':  mus_aro,
            }
            record['purchase_intent'] = np.nanmean([buy, interest, nobuy_r])
            record['memory_score'] = np.nanmean([memory, mem_mood, mem_world])
            records.append(record)

long_df = pd.DataFrame(records)
print(f"\nStep5: ロング形式変換完了（{len(long_df)}行）")
print(f"  WTP欠損値数: {long_df['wtp'].isna().sum()}件")

# ============================================
# Step6: 購買意欲スケールの平均算出 + Cronbach's α
# ============================================
if len(long_df) > 0:
    # 購買意欲のCronbach's α
    alpha_data = long_df[['buy', 'interest', 'nobuy_r']].dropna()
    if len(alpha_data) > 2:
        alpha_val, alpha_ci = pg.cronbach_alpha(data=alpha_data)
        print(f"\nStep6: 購買意欲 Cronbach's α = {alpha_val:.3f}（目標 ≥ 0.7）")
        print("  項目を1つ除外したときのα:")
        for col in ['buy', 'interest', 'nobuy_r']:
            sub = alpha_data.drop(columns=[col])
            a, _ = pg.cronbach_alpha(data=sub)
            print(f"    {col}を除外: α = {a:.3f}")

    # 余韻のCronbach's α
    mem_alpha_data = long_df[['memory', 'mem_mood', 'mem_world']].dropna()
    MEMORY_ALPHA_THRESHOLD = 0.7  # 目安。内容的妥当性も考慮して判断すること
    USE_MEMORY_SCORE = False  # αの結果で上書きされる

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
            print(f"\n  → α ≥ {MEMORY_ALPHA_THRESHOLD}：3問を一次元として平均（memory_score）を従属変数に使用")
        else:
            USE_MEMORY_SCORE = False
            print(f"\n  → α < {MEMORY_ALPHA_THRESHOLD}：3問を別々に分析（Bonferroni補正 α=0.017）")
            print("  ※ 内容的妥当性も踏まえて最終判断すること")

# ============================================
# Step7: 予備評定の確認
# ============================================
if len(long_df) > 0 and 'video_valence' in long_df.columns:
    print("\nStep7: 予備評定の確認")
    pos_vval = long_df[long_df['video_type']=='pos']['video_valence'].mean()
    mel_vval = long_df[long_df['video_type']=='mel']['video_valence'].mean()
    print(f"  映像valence: ポジ={pos_vval:.2f}, メラ={mel_vval:.2f}")
    print(f"  → {'✅ 意図通り（ポジ > メラ）' if pos_vval > mel_vval else '⚠ 要確認'}")

    con_mval = long_df[long_df['music_cond']=='con']['music_valence'].mean()
    dis_mval = long_df[long_df['music_cond']=='dis']['music_valence'].mean()
    print(f"  音楽valence: 調和={con_mval:.2f}, 不調和={dis_mval:.2f}")

# ============================================
# Step8: 記述統計
# ============================================
if len(long_df) > 0:
    print("\nStep8: 記述統計")
    dvs_desc = ['purchase_intent', 'memory', 'wtp', 'cogfit']

    desc = long_df.groupby(['video_type', 'music_cond'])[dvs_desc].agg(
        ['mean', 'std', 'min', 'max', 'count']
    )
    print(desc.to_string())

    # WTPの外れ値確認
    print(f"\n  WTP最大値: {long_df['wtp'].max():.0f}円")
    print(f"  WTP最小値: {long_df['wtp'].min():.0f}円")
    print(f"  WTP中央値: {long_df['wtp'].median():.0f}円")

# ============================================
# Step9: 分布の確認（ANOVA前）
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
    print("\nStep9: 分布確認グラフ保存 → distributions.png")

# ============================================
# Step10: 分析
# ============================================
print("\nStep10: 分析開始")

# α結果に応じて余韻の分析方針を決定
if USE_MEMORY_SCORE:
    # α ≥ 0.7：3問平均を使う
    dvs = {
        'purchase_intent': '購買意欲',
        'memory_score':    '余韻持続（3問平均）',
        'wtp':             'WTP',
        'cogfit':          'cognitive fit（操作チェック）',
    }
else:
    # α < 0.7：3問を個別に分析
    dvs = {
        'purchase_intent': '購買意欲',
        'memory':          '余韻（記憶）',
        'mem_mood':        '余韻（感情持続）',
        'mem_world':       '余韻（没入持続）',
        'wtp':             'WTP',
        'cogfit':          'cognitive fit（操作チェック）',
    }
    print("\n⚠ 余韻を3問個別に分析します。有意水準はBonferroni補正でα=0.017を適用。")

if len(long_df) > 0:
    for dv, label in dvs.items():
        print(f"\n{'='*40}")
        print(f"【{label}】")

        data = long_df[['participant_id', 'video_type', 'music_cond', dv]].dropna()

        if dv == 'cogfit':
            con = data[data['music_cond'] == 'con'].set_index('participant_id')[dv]
            dis = data[data['music_cond'] == 'dis'].set_index('participant_id')[dv]
            common = con.index.intersection(dis.index)
            t_stat, p_val = stats.ttest_rel(con[common], dis[common])
            print(f"  調和条件 M={con.mean():.2f}, 不調和条件 M={dis.mean():.2f}")
            print(f"  対応ありt検定: t={t_stat:.3f}, p={p_val:.3f} {'*' if p_val < 0.05 else 'n.s.'}")

        else:
            if dv == 'wtp':
                stat, p_norm = stats.shapiro(data[dv].dropna())
                print(f"  Shapiro-Wilk検定（記録用）: W={stat:.3f}, p={p_norm:.3f}")
                print(f"  → WTPは一律で対数変換（log(WTP+1)）を適用")
                data = data.copy()
                data[dv] = np.log1p(data[dv])
                print(f"  ※ ANOVAは対数変換後、グラフは変換前スコアで表示")

            try:
                aov = pg.rm_anova(
                    data=data,
                    dv=dv,
                    within=['video_type', 'music_cond'],
                    subject='participant_id',
                    detailed=True
                )
                print(aov[['Source', 'F', 'p-unc', 'np2']].to_string(index=False))

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
                            sig = '*' if p_val < 0.025 else 'n.s.'
                            print(f"    {label_v}: 調和M={con.mean():.2f} vs 不調和M={dis.mean():.2f}, p={p_val:.3f} {sig}")

            except Exception as e:
                print(f"  ANOVA実行エラー（データ不足の可能性）: {e}")

# ============================================
# Step11: 可視化（変換前スコアで表示）
# ============================================
if len(long_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # α結果に応じてmemory_scoreを含めるか決定
    plot_dvs    = ['purchase_intent', 'wtp']
    plot_labels = ['購買意欲', 'WTP（円）']
    if USE_MEMORY_SCORE:
        plot_dvs.insert(1, 'memory_score')
        plot_labels.insert(1, '余韻持続（3問平均）')

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
    print("\nStep11: 結果グラフ保存 → results.png")
# ============================================
# Step12: WTPアンカリング検証
# ============================================
print("\nStep12: WTPアンカリング検証")

if len(long_df) > 0 and 'wtp' in long_df.columns:

    # --- 分析① 混合効果モデル（試行順序 × 音楽条件）---
    # rm ANOVAを使わない理由：
    # 各参加者の各試行には調和か不調和どちらかのデータしかない（セルが不完全）
    # 2要因rm ANOVAは「各参加者が全セルのデータを持つ」ことが前提のため不適切
    # 混合効果モデルは不完全なセル構造でも対応できる
    #
    # 注意：WTPの変化はアンカリング以外に以下の可能性も排除できない
    # - 疲労効果：後半になるほど回答が雑になる
    # - 学習効果：実験に慣れてきて回答が安定してくる
    # したがって「アンカリングを含む順序効果の可能性を示唆する」探索的分析として位置づける
    print("\n--- ① 混合効果モデル（試行順序 × 音楽条件）（WTP） ---")
    print("  ※ 収束の原因はアンカリング・疲労効果・学習効果の可能性があり区別できない")
    print("  ※ 探索的補助分析として位置づける")

    wtp_data = long_df[['participant_id', 'trial', 'music_cond', 'wtp']].dropna()
    wtp_data = wtp_data.copy()
    wtp_data['wtp_log'] = np.log1p(wtp_data['wtp'])
    wtp_data['music_cond_num'] = (wtp_data['music_cond'] == 'con').astype(int)  # con=1, dis=0

    try:
        import statsmodels.formula.api as smf

        # 混合効果モデル
        # wtp_log ~ trial + music_cond + trial:music_cond + (1|participant_id)
        model = smf.mixedlm(
            "wtp_log ~ trial * music_cond_num",
            data=wtp_data,
            groups=wtp_data["participant_id"]
        )
        result = model.fit(reml=True)
        print(result.summary())

        # 交互作用係数の確認
        interaction_coef = result.params.get('trial:music_cond_num', None)
        interaction_p = result.pvalues.get('trial:music_cond_num', None)

        if interaction_p is not None:
            print(f"\n  交互作用（trial × music_cond）: β={interaction_coef:.4f}, p={interaction_p:.3f}")
            if interaction_p < 0.05:
                print("  → 試行順序によって音楽条件のWTPへの効果が変化している")
                print("  → アンカリングを含む順序効果の可能性と整合的")
                print("  ※ 疲労効果・学習効果等の可能性も排除できない")
            else:
                print("  → 試行順序による音楽条件の効果の変化は確認されなかった")
                print("  → 順序効果を示唆する明確なパターンは見られない")

    except Exception as e:
        print(f"  混合効果モデル実行エラー: {e}")

    # --- 分析② 試行1との差分（補助分析）---
    # 主分析（①）を補完する記述・可視化
    # 「WTPが実際にどのように変化したか」を直感的に確認する目的
    print("\n--- ② 試行1との差分（補助分析：WTPの変化を可視化） ---")
    print("  ※ 主分析①の補完。統計的検定ではなく記述・可視化が目的")

    # 試行1のWTPを取得
    trial1_wtp = long_df[long_df['trial'] == 1][['participant_id', 'music_cond', 'wtp']].copy()
    trial1_wtp = trial1_wtp.rename(columns={'wtp': 'wtp_trial1'})

    # 試行2〜4に試行1のWTPをマージ
    wtp_diff = long_df[long_df['trial'] > 1][['participant_id', 'trial', 'music_cond', 'wtp']].copy()
    wtp_diff = wtp_diff.merge(
        trial1_wtp[['participant_id', 'wtp_trial1']],
        on='participant_id',
        how='left'
    )
    wtp_diff['diff_from_trial1'] = wtp_diff['wtp'] - wtp_diff['wtp_trial1']

    # 差分の記述統計
    diff_summary = wtp_diff.groupby(['trial', 'music_cond'])['diff_from_trial1'].agg(
        ['mean', 'std']
    ).reset_index()
    print(diff_summary.to_string(index=False))

    # 差分の可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：試行ごとのWTP平均（調和/不調和）
    ax1 = axes[0]
    wtp_by_trial = long_df.groupby(['trial', 'music_cond'])['wtp'].mean().reset_index()
    for cond, color, label in zip(['con', 'dis'], ['#E07B54', '#5B8DB8'], ['調和', '不調和']):
        sub = wtp_by_trial[wtp_by_trial['music_cond'] == cond]
        ax1.plot(sub['trial'], sub['wtp'], marker='o', color=color, label=label, linewidth=2)
    ax1.set_title('試行ごとのWTP平均')
    ax1.set_xlabel('試行')
    ax1.set_ylabel('WTP（円）')
    ax1.set_xticks([1, 2, 3, 4])
    ax1.legend()

    # 右：試行1との差分
    ax2 = axes[1]
    for cond, color, label in zip(['con', 'dis'], ['#E07B54', '#5B8DB8'], ['調和', '不調和']):
        sub = diff_summary[diff_summary['music_cond'] == cond]
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
    print("差分が明確に収束しない → アンカリングを示唆する明確なパターンは確認できない")
    print("※ いずれもアンカリングの有無を直接証明するものではない")
    print("※ 主分析①（混合効果モデル）と合わせて解釈すること")

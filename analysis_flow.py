# ============================================
# 特別演習I 分析フロー
# 映像種類(2) × 音楽条件(2) Two-way ANOVA
# ============================================

# ============================================
# Step1: データ読み込み
# ============================================
import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAexGothic'  # 日本語フォント

# GoogleスプレッドシートからCSVとしてエクスポートして読み込む
# df = pd.read_csv('responses.csv')
# 以下はテスト用ダミーデータで動作確認
df = pd.DataFrame()  # 実際はCSV読み込みに差し替え

print("Step1: データ読み込み完了")
print(f"参加者数: {len(df)}")

# ============================================
# Step2: グループIDから条件ラベル付与
# ============================================

# 8グループ×4試行のマッピングテーブル
# 各セルは (映像種類, 音楽条件)
# 映像種類: pos=ポジティブ, mel=メランコリック
# 音楽条件: con=調和, dis=不調和

GROUP_MAP = {
    #         試行1              試行2              試行3              試行4
    'G1a': [('pos','con'), ('mel','dis'), ('pos','dis'), ('mel','con')],  # A  C' B' D
    'G1b': [('mel','con'), ('pos','dis'), ('mel','dis'), ('pos','con')],  # C  A' D' B
    'G2a': [('pos','con'), ('mel','dis'), ('pos','dis'), ('mel','con')],  # B  D' A' C
    'G2b': [('mel','con'), ('pos','dis'), ('mel','dis'), ('pos','con')],  # D  B' C' A
    'G3a': [('pos','dis'), ('mel','con'), ('pos','con'), ('mel','dis')],  # A' C  B  D'
    'G3b': [('mel','dis'), ('pos','con'), ('mel','con'), ('pos','dis')],  # C' A  D  B'
    'G4a': [('pos','dis'), ('mel','con'), ('pos','con'), ('mel','dis')],  # B' D  A  C'
    'G4b': [('mel','dis'), ('pos','con'), ('mel','con'), ('pos','dis')],  # D' B  C  A'
}

# フォームの列名（GASで生成したフォームに合わせて調整）
TRIAL_COLS = {
    1: {
        'memory':   '[試行1] この映像のことが頭に残っている',
        'buy':      '[試行1] この商品を購入したいと思う',
        'interest': '[試行1] この商品に興味がある',
        'nobuy':    '[試行1] この商品を買う気はしない',
        'wtp':      '[試行1] この商品にいくらまで払えますか？',
        'cogfit':   '[試行1] この映像と音楽は雰囲気が合っていた',
        'attention': None,  # 試行1にはattention checkなし
    },
    2: {
        'memory':   '[試行2] この映像のことが頭に残っている',
        'buy':      '[試行2] この商品を購入したいと思う',
        'interest': '[試行2] この商品に興味がある',
        'nobuy':    '[試行2] この商品を買う気はしない',
        'wtp':      '[試行2] この商品にいくらまで払えますか？',
        'cogfit':   '[試行2] この映像と音楽は雰囲気が合っていた',
        'attention': 'この質問では4を選択してください',
    },
    3: {
        'memory':   '[試行3] この映像のことが頭に残っている',
        'buy':      '[試行3] この商品を購入したいと思う',
        'interest': '[試行3] この商品に興味がある',
        'nobuy':    '[試行3] この商品を買う気はしない',
        'wtp':      '[試行3] この商品にいくらまで払えますか？',
        'cogfit':   '[試行3] この映像と音楽は雰囲気が合っていた',
        'attention': None,
    },
    4: {
        'memory':   '[試行4] この映像のことが頭に残っている',
        'buy':      '[試行4] この商品を購入したいと思う',
        'interest': '[試行4] この商品に興味がある',
        'nobuy':    '[試行4] この商品を買う気はしない',
        'wtp':      '[試行4] この商品にいくらまで払えますか？',
        'cogfit':   '[試行4] この映像と音楽は雰囲気が合っていた',
        'attention': None,
    },
}

print("Step2: グループIDから条件ラベル付与")

# ============================================
# Step3: 逆転項目の処理
# ============================================
# 「この商品を買う気はしない」を反転（9 - スコア）
for t in [1, 2, 3, 4]:
    col = TRIAL_COLS[t]['nobuy']
    if col in df.columns:
        df[col] = 9 - df[col]

print("Step3: 逆転項目の処理完了")

# ============================================
# Step4: attention checkで除外
# ============================================
# 正解は4（UI編集で設定した値に合わせて変更）
ATTENTION_ANSWER = 4
before = len(df)
excluded = []

for t in [1, 2, 3, 4]:
    col = TRIAL_COLS[t].get('attention')
    if col and col in df.columns:
        mask = df[col] != ATTENTION_ANSWER
        excluded.extend(df[mask].index.tolist())

excluded = list(set(excluded))
df = df.drop(index=excluded)
print(f"Step4: attention check除外 {len(excluded)}名 → 残り{len(df)}名")

# ============================================
# Step5: 購買意欲スケールの平均算出 + Cronbach's α
# ============================================
# ロング形式に変換（1行=1試行）
records = []
for idx, row in df.iterrows():
    group = row.get('グループID（実験者から指定されたものを選んでください）', '')
    if group not in GROUP_MAP:
        continue
    for t in [1, 2, 3, 4]:
        video_type, music_cond = GROUP_MAP[group][t-1]
        cols = TRIAL_COLS[t]
        record = {
            'participant_id': idx,
            'group':          group,
            'trial':          t,
            'video_type':     video_type,   # pos / mel
            'music_cond':     music_cond,   # con / dis
            'memory':         row.get(cols['memory'],   np.nan),
            'buy':            row.get(cols['buy'],      np.nan),
            'interest':       row.get(cols['interest'], np.nan),
            'nobuy_r':        row.get(cols['nobuy'],    np.nan),  # 逆転済み
            'wtp':            row.get(cols['wtp'],      np.nan),
            'cogfit':         row.get(cols['cogfit'],   np.nan),
        }
        # 購買意欲3問の平均
        record['purchase_intent'] = np.nanmean([
            record['buy'], record['interest'], record['nobuy_r']
        ])
        records.append(record)

long_df = pd.DataFrame(records)

# Cronbach's α（購買意欲3問）
alpha_data = long_df[['buy', 'interest', 'nobuy_r']].dropna()
if len(alpha_data) > 0:
    alpha_result = pg.cronbach_alpha(data=alpha_data)
    print(f"Step5: Cronbach's α = {alpha_result[0]:.3f} (目標 ≥ 0.7)")

# ============================================
# Step6: Two-way ANOVA
# ============================================
print("\nStep6: Two-way ANOVA")
print("従属変数: purchase_intent / memory / wtp / cogfit")

dvs = {
    'purchase_intent': '購買意欲',
    'memory':          '余韻持続',
    'wtp':             'WTP',
    'cogfit':          'cognitive fit（操作チェック）',
}

for dv, label in dvs.items():
    print(f"\n--- {label} ---")
    data = long_df[['participant_id', 'video_type', 'music_cond', dv]].dropna()

    if dv == 'cogfit':
        # cognitive fitは操作チェック：調和vs不調和のt検定
        con = data[data['music_cond'] == 'con'][dv]
        dis = data[data['music_cond'] == 'dis'][dv]
        t_stat, p_val = stats.ttest_ind(con, dis)
        print(f"  調和条件 M={con.mean():.2f}, 不調和条件 M={dis.mean():.2f}")
        print(f"  t={t_stat:.3f}, p={p_val:.3f} {'*' if p_val < 0.05 else 'n.s.'}")
    else:
        # WTPは正規性確認
        if dv == 'wtp':
            stat, p_norm = stats.shapiro(data[dv])
            print(f"  正規性検定（Shapiro-Wilk）: p={p_norm:.3f} {'→対数変換推奨' if p_norm < 0.05 else '→正規性OK'}")
            if p_norm < 0.05:
                data = data.copy()
                data[dv] = np.log1p(data[dv])  # log(WTP+1)

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

            # 交互作用が有意なら単純主効果
            interaction_p = aov[aov['Source'] == 'video_type * music_cond']['p-unc'].values
            if len(interaction_p) > 0 and interaction_p[0] < 0.05:
                print("  → 交互作用有意：各映像種類内で音楽条件を比較")
                for vtype in ['pos', 'mel']:
                    sub = data[data['video_type'] == vtype]
                    con = sub[sub['music_cond'] == 'con'][dv]
                    dis = sub[sub['music_cond'] == 'dis'][dv]
                    t_stat, p_val = stats.ttest_rel(con, dis)
                    print(f"    {vtype}: 調和M={con.mean():.2f} vs 不調和M={dis.mean():.2f}, p={p_val:.3f} {'*' if p_val < 0.05 else 'n.s.'}")
        except Exception as e:
            print(f"  ANOVA実行エラー（データ不足の可能性）: {e}")

# ============================================
# Step7: 可視化
# ============================================
print("\nStep7: 可視化")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plot_dvs = ['purchase_intent', 'memory', 'wtp']
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
    ax.set_ylabel('平均スコア')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['調和', '不調和'])
    ax.legend()

plt.tight_layout()
plt.savefig('results.png', dpi=150)
plt.show()
print("グラフ保存: results.png")

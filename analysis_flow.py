# ============================================
# 特別演習I 分析フロー v15
# 映像valence(2) × 音楽valence(2) Two-way repeated measures ANOVA
# ============================================
# v15の変更点（9/27）
#   1. 事前の感情測定（baseline mood / arousal）を追加
#      - Step5 で participant単位の値として保持
#      - Step7-0 として操作チェックの補強に使用
#        （映像視聴後のvalence/arousalとの差分で感情誘発の方向と大きさを確認）
#      - Step7-1 で各従属変数との相関を探索的に確認
#      - ANCOVAによる統制は行わない（被験者内計画で個人差は既に除去済み、
#        N=24で自由度を消費すると検出力が落ちるため）
#
# v14の変更点
#   1. GROUP_MAP を修正（v13は試行2・4の条件が誤っていた）
#   2. MATERIAL_ORDER を追加（実験実施時にどのmp4を流すかの対応表）
#   3. verify_group_map() を追加（実行時に設計要件を自己検証）
#   4. 余韻と購買意欲の相関を Step7-1 に追加（9/24決定：探索的に検討）
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

def get_col(df, keyword):
    """試行番号を持たない列（基本属性・事前測定など）をキーワードで取得"""
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
    print("\n正規化後の列名一覧:")
    for c in df.columns:
        print(f"  {c}")

# ============================================
# Step2: グループIDから条件ラベル付与の定義
# ============================================
# 実験デザイン：映像valence × 音楽valenceの2×2
#   congruent   = 映像と音楽のvalenceが一致
#   incongruent = 映像と音楽のvalenceが不一致
#
# 条件記号（研究計画書の「条件の記号」表に対応）
#   A / A' = ポジティブ①（滝）         ／ 調和・不調和
#   B / B' = ポジティブ②（夕日）       ／ 調和・不調和
#   C / C' = メランコリック①（ベンチ） ／ 調和・不調和
#   D / D' = メランコリック②（雨林）   ／ 調和・不調和
#
# 注意：分析上はどの映像素材かを区別しない（valenceのみを扱う）ため、
# G1aとG2a、G1bとG2b、G3aとG4a、G3bとG4b は同一のタプル列になる。
# これは設計通りであり誤りではない。8グループに分けているのは
# 各条件を各順番位置に均等に配置するためであり、使用する素材の違いは
# 下の MATERIAL_ORDER が保持している。

GROUP_MAP = {
    #         試行1                  試行2                  試行3                  試行4         記号
    'G1a': [('pos','pos','con'), ('mel','pos','inc'), ('pos','mel','inc'), ('mel','mel','con')],  # A   C'  B'  D
    'G1b': [('mel','mel','con'), ('pos','mel','inc'), ('mel','pos','inc'), ('pos','pos','con')],  # C   A'  D'  B
    'G2a': [('pos','pos','con'), ('mel','pos','inc'), ('pos','mel','inc'), ('mel','mel','con')],  # B   D'  A'  C
    'G2b': [('mel','mel','con'), ('pos','mel','inc'), ('mel','pos','inc'), ('pos','pos','con')],  # D   B'  C'  A
    'G3a': [('pos','mel','inc'), ('mel','mel','con'), ('pos','pos','con'), ('mel','pos','inc')],  # A'  C   B   D'
    'G3b': [('mel','pos','inc'), ('pos','pos','con'), ('mel','mel','con'), ('pos','mel','inc')],  # C'  A   D   B'
    'G4a': [('pos','mel','inc'), ('mel','mel','con'), ('pos','pos','con'), ('mel','pos','inc')],  # B'  D   A   C'
    'G4b': [('mel','pos','inc'), ('pos','pos','con'), ('mel','mel','con'), ('pos','mel','inc')],  # D'  B   C   A'
}

# 実験実施用：どの素材（mp4）をどの順で流すか
# 分析には使わないが、実施時の確認と GROUP_MAP の照合に用いる
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
    """
    pingouinのバージョン差を吸収して、p値と効果量の列名を返す。
    0.5系: 'p-unc' / 'np2'
    0.6系: 'p_unc' / 'ng2'（2要因のときは一般化イータ二乗）
    """
    p_col = next((c for c in ['p-unc', 'p_unc', 'p-GG-corr', 'p_GG_corr'] if c in aov.columns), None)
    e_col = next((c for c in ['np2', 'ng2', 'n2'] if c in aov.columns), None)
    return p_col, e_col


def get_source_p(aov, source_name, p_col):
    """Source列が完全一致する行のp値を返す（無ければNone）"""
    hit = aov[aov['Source'] == source_name]
    if len(hit) == 0 or p_col is None:
        return None
    return float(hit[p_col].values[0])


def get_interaction_p(aov, p_col):
    """交互作用行のp値を返す。Source名は 'A * B' の形式"""
    hit = aov[aov['Source'].str.contains(r'\*', regex=True)]
    if len(hit) == 0 or p_col is None:
        return None
    return float(hit[p_col].values[0])


def verify_group_map():
    """
    GROUP_MAP が設計要件を満たすかを実行時に検証する。
    MATERIAL_ORDER から導出した条件と GROUP_MAP を照合するため、
    どちらかを編集して片方を直し忘れた場合に検出できる。
    """
    VIDEO_VAL = {'A': 'pos', 'B': 'pos', 'C': 'mel', 'D': 'mel'}

    def decode(sym):
        base, is_inc = sym[0], sym.endswith("'")
        v = VIDEO_VAL[base]
        m = ('pos' if v == 'mel' else 'mel') if is_inc else v
        return (v, m, 'inc' if is_inc else 'con')

    all_ok = True

    # 照合：MATERIAL_ORDER から導出した条件と GROUP_MAP が一致するか
    for g, syms in MATERIAL_ORDER.items():
        derived = [decode(s) for s in syms]
        if derived != GROUP_MAP[g]:
            print(f"  ✗ {g}: GROUP_MAPとMATERIAL_ORDERが不一致")
            print(f"      GROUP_MAP    : {GROUP_MAP[g]}")
            print(f"      MATERIAL由来 : {derived}")
            all_ok = False

    # 各グループの設計要件
    for g, conds in GROUP_MAP.items():
        n_con = sum(1 for c in conds if c[2] == 'con')
        vids = [c[0] for c in conds]
        alt = all(vids[i] != vids[i + 1] for i in range(3))
        bases = [s[0] for s in MATERIAL_ORDER[g]]
        uniq = len(set(bases)) == 4
        if not (n_con == 2 and alt and uniq):
            print(f"  ✗ {g}: 調和{n_con}回 / ポジメラ交互={alt} / 映像4種={uniq}")
            all_ok = False

    # 各試行位置で congruency と video valence が均等か
    for t in range(4):
        cs = [GROUP_MAP[g][t][2] for g in GROUP_MAP]
        vs = [GROUP_MAP[g][t][0] for g in GROUP_MAP]
        if cs.count('con') != 4 or vs.count('pos') != 4:
            print(f"  ✗ 試行{t+1}: congruency con={cs.count('con')} / video pos={vs.count('pos')}")
            all_ok = False

    if all_ok:
        print("Step2: GROUP_MAP検証 → ✅ 全要件を満たしています")
        print("  各グループ：調和2回・不調和2回／ポジとメラが交互／映像4種を1回ずつ")
        print("  各試行位置：congruency・video valenceともに4対4で均等")
    else:
        print("Step2: ⚠ GROUP_MAPに問題があります。実験実施前に必ず修正すること")
    return all_ok


verify_group_map()

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

    # 事前の感情測定（実験開始前・参加者ごとに1つ）
    base_mood_col = get_col(df, '今の気分')
    base_aro_col  = get_col(df, '今の興奮状態')

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
                'material':       MATERIAL_ORDER[group][t - 1],  # どの素材を使ったか（記録用）
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
                # cogfit: 変数名は歴史的経緯によるもの
                # 実際に測定しているのは perceived congruency（知覚された一致度）
                # Vessey & Galletta (1991) の cognitive fit 理論とは対象・測定方法が異なる
                'cogfit':         cogfit,
                'vid_val_check':  vid_val,     # 操作チェック用
                'vid_aro_check':  vid_aro,
                'mus_val_check':  mus_val,
                'mus_aro_check':  mus_aro,
                'liking':         liking,      # 映像への好意度（交絡変数）
                # 事前の感情測定（全試行で同じ値。参加者のベースライン）
                'base_mood':      base_mood,
                'base_arousal':   base_aro,
            }
            # 映像視聴後の評定とベースラインの差分（感情誘発の方向と大きさ）
            record['vid_val_shift'] = (vid_val - base_mood) if pd.notna(vid_val) and pd.notna(base_mood) else np.nan
            record['vid_aro_shift'] = (vid_aro - base_aro)  if pd.notna(vid_aro) and pd.notna(base_aro)  else np.nan
            record['purchase_intent'] = np.nanmean([buy, interest, nobuy_r])
            records.append(record)

long_df = pd.DataFrame(records)
print(f"\nStep5: ロング形式変換完了（{len(long_df)}行）")
if len(long_df) > 0:
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
# Step7-0: 事前の感情測定と感情誘発の確認
# ============================================
# 実験開始前に測定したベースライン（気分・興奮状態）を報告し、
# 映像視聴後の評定との差分から感情誘発の方向と大きさを確認する。
#
# 位置づけ：操作チェックの補強
#   Step7の「ポジ映像 vs メラ映像」の比較はベースラインを持たないため、
#   「どちらの方向にどれだけ動いたか」が言えない。
#   事前測定との差分を見ることで、感情誘発が実際に起きたかを直接示せる。
#
# ANCOVAによる統制は行わない：
#   被験者内計画のため個人差はrm ANOVAで既に除去されており、
#   N=24で共変量を入れると自由度を消費して検出力がさらに落ちるため。
if len(long_df) > 0 and 'base_mood' in long_df.columns:
    print("\nStep7-0: 事前の感情測定と感情誘発の確認")

    # ベースラインの記述統計（参加者単位）
    base = long_df.groupby('participant_id')[['base_mood', 'base_arousal']].first()
    print(f"  事前の気分     : M={base['base_mood'].mean():.2f}, SD={base['base_mood'].std():.2f}, "
          f"range={base['base_mood'].min():.0f}-{base['base_mood'].max():.0f}")
    print(f"  事前の興奮状態 : M={base['base_arousal'].mean():.2f}, SD={base['base_arousal'].std():.2f}, "
          f"range={base['base_arousal'].min():.0f}-{base['base_arousal'].max():.0f}")

    # 感情誘発の方向と大きさ（ベースラインからの変化量）
    print("\n  映像視聴後のベースラインからの変化量:")
    for vtype, lab in [('pos', 'ポジティブ映像'), ('mel', 'メランコリック映像')]:
        sub = long_df[long_df['video_valence'] == vtype]
        dv_ = sub['vid_val_shift'].mean()
        da_ = sub['vid_aro_shift'].mean()
        print(f"    {lab}: valence {dv_:+.2f} / arousal {da_:+.2f}")

    pos_shift = long_df[long_df['video_valence'] == 'pos']['vid_val_shift'].mean()
    mel_shift = long_df[long_df['video_valence'] == 'mel']['vid_val_shift'].mean()
    if pd.notna(pos_shift) and pd.notna(mel_shift):
        ok_pos = pos_shift > 0
        ok_mel = mel_shift < 0
        print(f"\n    ポジ映像でvalence上昇: {'✅' if ok_pos else '⚠ 上昇していない'}")
        print(f"    メラ映像でvalence下降: {'✅' if ok_mel else '⚠ 下降していない'}")
        if ok_pos and ok_mel:
            print("    → 感情誘発は意図した方向に働いたと考えられる")
        else:
            print("    → 感情誘発が意図通りでない可能性。limitationsに記載すること")

    print("\n  ※ ベースラインは全試行で共通のため、条件間の比較には影響しない")
    print("  ※ 差分は記述的な確認であり、統計的検定は行わない")

# ============================================
# Step7: 操作チェック（valence/arousal）
# ============================================
if len(long_df) > 0:
    print("\nStep7: 操作チェック")

    # 映像valenceの確認：ポジ映像 > メラ映像になっているか
    pos_vval = long_df[long_df['video_valence'] == 'pos']['vid_val_check'].mean()
    mel_vval = long_df[long_df['video_valence'] == 'mel']['vid_val_check'].mean()
    print(f"  映像valence: ポジ={pos_vval:.2f}, メラ={mel_vval:.2f}")
    print(f"  → {'✅ 意図通り（ポジ > メラ）' if pos_vval > mel_vval else '⚠ 要確認'}")

    # 音楽valenceの確認：ポジ音楽 > メラ音楽になっているか
    pos_mval = long_df[long_df['music_valence'] == 'pos']['mus_val_check'].mean()
    mel_mval = long_df[long_df['music_valence'] == 'mel']['mus_val_check'].mean()
    print(f"  音楽valence: ポジ={pos_mval:.2f}, メラ={mel_mval:.2f}")
    print(f"  → {'✅ 意図通り（ポジ > メラ）' if pos_mval > mel_mval else '⚠ 要確認'}")

    # perceived congruency（知覚された一致度）の操作チェック：congruent > incongruent
    # ※これは仮説検証ではなく操作が知覚されたかの確認であるため、
    #   Step10の仮説検証における多重比較の補正対象には含めない（α=.05のまま）
    # 注意：perceived congruencyで差が出ることは両義的
    #   (1) 操作が知覚された証拠（操作成功）
    #   (2) 参加者が条件操作の存在に気づいていた証拠（demand effectのリスク）
    #   参加者は演習の実験協力という文脈のため操作に気づく可能性が高い
    #   この両義性をlimitationsに明記する
    con_cf = long_df[long_df['congruency'] == 'con']['cogfit'].mean()
    inc_cf = long_df[long_df['congruency'] == 'inc']['cogfit'].mean()
    con_data = long_df[long_df['congruency'] == 'con'].set_index('participant_id')['cogfit']
    inc_data = long_df[long_df['congruency'] == 'inc'].set_index('participant_id')['cogfit']
    common = con_data.index.intersection(inc_data.index)
    if len(common) > 1:
        t_stat, p_val = stats.ttest_rel(con_data[common], inc_data[common])
        print(f"\n  perceived congruency操作チェック:")
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
    pos_varo = long_df[long_df['video_valence'] == 'pos']['vid_aro_check'].mean()
    mel_varo = long_df[long_df['video_valence'] == 'mel']['vid_aro_check'].mean()
    print(f"\n  映像arousal（探索的）: ポジ={pos_varo:.2f}, メラ={mel_varo:.2f}")

# ============================================
# Step7-1: 余韻と購買意欲の関係（探索的検討）
# ============================================
# 本研究の仮説は2つに分けている
#   仮説1：congruentな音楽環境では余韻が持続する
#   仮説2：congruentな音楽環境では購買意欲が高い
# 「余韻が購買意欲を生む」という媒介関係は仮説として立てておらず、
# 本Stepでは両者の相関を記述的に報告するにとどめる。
# N=24では媒介分析の検出力が不足するため、媒介の検証は今後の課題とする。
if len(long_df) > 0:
    print("\nStep7-1: 余韻と購買意欲の関係（探索的検討）")

    mem_col = 'memory_score' if USE_MEMORY_SCORE else None

    if mem_col:
        pair = long_df[[mem_col, 'purchase_intent']].dropna()
        if len(pair) > 2:
            r_mp = pair.corr().loc[mem_col, 'purchase_intent']
            print(f"  余韻（3問平均） × 購買意欲: r={r_mp:.3f}（n={len(pair)}試行）")
    else:
        print("  ※ 余韻のαが基準を下回ったため、3項目それぞれと購買意欲の相関を出す")
        for col, lab in [('memory', '記憶'), ('mem_mood', '感情持続'), ('mem_world', '没入持続')]:
            pair = long_df[[col, 'purchase_intent']].dropna()
            if len(pair) > 2:
                r_mp = pair.corr().loc[col, 'purchase_intent']
                print(f"  余韻（{lab}） × 購買意欲: r={r_mp:.3f}（n={len(pair)}試行）")

    # WTPとの関係も併せて記述
    if mem_col:
        pair_w = long_df[[mem_col, 'wtp']].dropna()
        if len(pair_w) > 2:
            r_mw = pair_w.corr().loc[mem_col, 'wtp']
            print(f"  余韻（3問平均） × WTP:     r={r_mw:.3f}（n={len(pair_w)}試行）")

    print("  ※ これは記述的な相関であり、媒介関係を示すものではない")
    print("  ※ 同一参加者の複数試行を含むため、独立性の仮定は満たしていない")

    # --- 事前の感情と各従属変数の相関（探索的）---
    # 実験前の気分が良かった人ほど購買意欲が高いか等を確認する。
    # 感情ヒューリスティック（Slovic et al., 2002）の枠組みと整合するかの傍証。
    if 'base_mood' in long_df.columns:
        print("\n  事前の感情と各従属変数の相関（探索的）:")
        dv_list = [('purchase_intent', '購買意欲'), ('wtp', 'WTP')]
        if USE_MEMORY_SCORE:
            dv_list.insert(1, ('memory_score', '余韻'))
        for base_col, base_lab in [('base_mood', '事前の気分'), ('base_arousal', '事前の興奮状態')]:
            for dv_col, dv_lab in dv_list:
                pair_b = long_df[[base_col, dv_col]].dropna()
                if len(pair_b) > 2 and pair_b[base_col].nunique() > 1:
                    r_b = pair_b.corr().loc[base_col, dv_col]
                    print(f"    {base_lab} × {dv_lab}: r={r_b:.3f}")
        print("    ※ ベースラインは参加者単位の値のため、試行単位の相関は")
        print("       同一参加者のデータが繰り返し含まれる点に注意")

# ============================================
# Step7-2: demand認知チェック
# ============================================
# 参加者が実験の意図に気づいていたかを確認する
if len(df) > 0:
    print("\nStep7-2: demand認知チェック")

    demand_col = [c for c in df.columns if '実験の目的' in c]
    if demand_col:
        demand_col = demand_col[0]
        print(f"  回答一覧（目視確認が必要）:")
        for idx, ans in df[demand_col].items():
            print(f"    参加者{idx}: {ans}")

        # 判定1：実験目的の推測（音楽・購買への言及）
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

        # 判定2：条件操作への気づき（意図的な不調和への言及）
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
        print("  ※ 除外するかどうかは目視確認後に判断する")
        print("  ※ 参加者は演習の実験協力という文脈のため操作に気づく可能性が高い")
        print("  ※ 気づいた参加者の割合を報告し、limitationsに記載する")
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

    # 事前の感情測定（参加者単位）
    if 'base_mood' in long_df.columns:
        base_desc = long_df.groupby('participant_id')[['base_mood', 'base_arousal']].first()
        print("\n  事前の感情測定（参加者単位）:")
        print(base_desc.describe().loc[['mean', 'std', 'min', 'max']].to_string())

    # グループごとの人数（設計通り割り振れているかの確認）
    print("\n  グループごとの参加者数:")
    gcount = long_df.groupby('group')['participant_id'].nunique()
    for g in ['G1a','G1b','G2a','G2b','G3a','G3b','G4a','G4b']:
        n = gcount.get(g, 0)
        print(f"    {g}: {n}名")
    if gcount.nunique() > 1:
        print("  ⚠ グループ間で人数が不均等です。カウンターバランスが崩れている可能性あり")

# ============================================
# Step9: 分布の確認（ANOVA前）
# ============================================
if len(long_df) > 0:
    plot_cols = ['purchase_intent', 'wtp', 'cogfit']
    plot_labels_hist = ['購買意欲', 'WTP（円）', 'perceived congruency']
    if USE_MEMORY_SCORE:
        plot_cols.insert(1, 'memory_score')
        plot_labels_hist.insert(1, '余韻持続（3問平均）')

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

        # rm_anovaは全セルが揃った参加者のみで実行する必要がある
        # WTPの欠損等で一部の試行が欠けた参加者は除外する
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
            # pingouinのバージョンによって列名が異なるため解決してから使う
            p_col, e_col = resolve_aov_cols(aov)
            show_cols = ['Source', 'F'] + [c for c in [p_col, e_col] if c]
            print(aov[show_cols].to_string(index=False))
            if e_col == 'ng2':
                print("  ※ 効果量は一般化イータ二乗（ng2）")

            # Step1：音楽valenceの主効果（仮説の直接検証）
            music_p = get_source_p(aov, 'music_valence', p_col)
            if music_p is not None:
                pos_m = data[data['music_valence'] == 'pos'][dv].mean()
                mel_m = data[data['music_valence'] == 'mel'][dv].mean()
                sig = '*' if music_p < 0.05 else 'n.s.'
                print(f"\n  【Step1】音楽valenceの主効果: p={music_p:.3f} {sig}")
                print(f"    ポジ音楽 M={pos_m:.2f} vs メラ音楽 M={mel_m:.2f}")

            # Step2：映像valenceの主効果（副次的知見）
            video_p = get_source_p(aov, 'video_valence', p_col)
            if video_p is not None:
                pos_v = data[data['video_valence'] == 'pos'][dv].mean()
                mel_v = data[data['video_valence'] == 'mel'][dv].mean()
                sig = '*' if video_p < 0.05 else 'n.s.'
                print(f"\n  【Step2】映像valenceの主効果: p={video_p:.3f} {sig}")
                print(f"    ポジ映像 M={pos_v:.2f} vs メラ映像 M={mel_v:.2f}")

            # Step3：交互作用（= congruencyの効果に相当）
            interaction_p = get_interaction_p(aov, p_col)
            if interaction_p is not None:
                sig = '*' if interaction_p < 0.05 else 'n.s.'
                print(f"\n  【Step3】交互作用（映像valence × 音楽valence）: p={interaction_p:.3f} {sig}")

                # 参考：congruent / incongruent の平均
                con_m = data[data['congruency'] == 'con'][dv].mean()
                inc_m = data[data['congruency'] == 'inc'][dv].mean()
                print(f"    参考 congruent M={con_m:.2f} vs incongruent M={inc_m:.2f}")

                # Step4：交互作用が有意なら単純主効果（Bonferroni補正：α=0.025）
                # 2回比較するため .05/2 = .025
                if interaction_p < 0.05:
                    print(f"  → 音楽valenceの効果が映像valenceによって異なる可能性")
                    print(f"  【Step4】単純主効果（Bonferroni補正 α=0.025）")
                    print(f"  ※ 2種類の不調和（メラ映像×ポジ音楽／ポジ映像×メラ音楽）が")
                    print(f"     同じ効果を持つとは限らない。差の大きさの違いに注目する")
                    for vtype in ['pos', 'mel']:
                        label_v = 'ポジティブ映像' if vtype == 'pos' else 'メランコリック映像'
                        sub = data[data['video_valence'] == vtype]
                        pos_mus = sub[sub['music_valence'] == 'pos'].set_index('participant_id')[dv]
                        mel_mus = sub[sub['music_valence'] == 'mel'].set_index('participant_id')[dv]
                        common = pos_mus.index.intersection(mel_mus.index)
                        if len(common) > 1:
                            t_stat, p_val = stats.ttest_rel(pos_mus[common], mel_mus[common])
                            sig = '*' if p_val < 0.025 else 'n.s.'
                            diff = pos_mus.mean() - mel_mus.mean()
                            print(f"    {label_v}: ポジ音楽M={pos_mus.mean():.2f} vs メラ音楽M={mel_mus.mean():.2f}"
                                  f" (差={diff:+.2f}), p={p_val:.3f} {sig}")

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

    fig, axes = plt.subplots(1, len(plot_dvs), figsize=(5 * len(plot_dvs), 5))
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
    # 注意：WTPの変化はアンカリング以外に疲労効果・学習効果の可能性も排除できない
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

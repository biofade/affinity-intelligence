# -*- coding: utf-8 -*-
"""
app.py -- Affinity Intelligence 本地演示前端 (Streamlit)

启动:
    cd F:\\DSH_Galatea\\Affinity_Intelligence
    streamlit run app.py

设计要点 (按需求):
  * 纯 Python + Streamlit, 无自定义 CSS, 无 React/Vue。
  * **不调用 subprocess**: 直接 import predict_affinity.get_prediction, 复用同一进程。
    (实际上 predict_affinity.py 里就是那个函数, 不需要 subprocess, 也没有启动开销)
  * 上传的 PDB 写入 NamedTemporaryFile, 预测后立刻删除, 不污染 data/pdb。
  * 模型/训练集统计/特征提取器都带缓存, 第二次点击起几乎瞬时。
"""

import os
import sys
import tempfile
import traceback

import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "scripts"))

from predict_affinity import (  # noqa: E402
    DEFAULT_MODEL,
    get_prediction,
    list_existing_pdbs,
    load_model_bundle,
)

MODEL_PATH = DEFAULT_MODEL

st.set_page_config(page_title="Affinity Intelligence 演示", layout="wide")
st.title("Affinity Intelligence — 蛋白-蛋白结合亲和力预测")
st.caption("LightGBM + 物理界面特征 | 家族隔离划分 | V1.1 模型")


# ---------------------------------------------------------------- 缓存
@st.cache_resource(show_spinner=False)
def _warm_up(model_path: str, mtime: float):
    """预热模型与特征提取器 (按模型修改时间缓存)。"""
    load_model_bundle(model_path)
    return True


try:
    _warm_up(MODEL_PATH, os.path.getmtime(MODEL_PATH))
except Exception as exc:                                        # noqa: BLE001
    st.error("模型加载失败: %s" % exc)
    st.stop()


# ---------------------------------------------------------------- 侧栏: 模型信息
with st.sidebar:
    st.header("模型信息")
    try:
        payload, stats = load_model_bundle(MODEL_PATH)
        st.write("模型文件: `%s`" % os.path.basename(MODEL_PATH))
        st.write("训练样本: **%s** 行" % payload.get("train_rows"))
        st.write("训练集统计: %s 行" % stats.get("train_n"))
        if stats.get("slope") is not None:
            st.write("拟合斜率: **%.2f**" % stats["slope"])
        if stats.get("rmse"):
            st.write("测试集 RMSE: **%.2f** kcal/mol" % stats["rmse"])
        st.divider()
        st.caption("划分策略: 40% 序列同一性家族级, 整族分侧 (无同源泄漏)")
    except Exception as exc:                                    # noqa: BLE001
        st.warning("模型信息读取失败: %s" % str(exc)[:80])


# ---------------------------------------------------------------- 输入区
st.subheader("1) 选择结构")

mode = st.radio("结构来源", ("从 data/pdb 目录选择", "上传 PDB 文件"),
                horizontal=True, index=0)

tmp_path = None          # 上传文件对应的临时路径, 用完必须删除
pdb_path = None

if mode == "从 data/pdb 目录选择":
    files = list_existing_pdbs()
    if not files:
        st.error("data/pdb 目录为空; 请改用上传方式。")
        st.stop()
    default_idx = files.index("1A22.pdb") if "1A22.pdb" in files else 0
    choice = st.selectbox("现有结构 (%d 个)" % len(files), files, index=default_idx)
    pdb_path = os.path.join(HERE, "data", "pdb", choice)
else:
    up = st.file_uploader("上传 PDB / CIF 文件", type=["pdb", "ent", "cif", "mmcif"])
    if up is not None:
        suffix = os.path.splitext(up.name)[1] or ".pdb"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                tf.write(up.getbuffer())
                tmp_path = tf.name
            pdb_path = tmp_path
            st.success("已接收: %s (%.1f KB)" % (up.name, len(up.getbuffer()) / 1024))
        except Exception as exc:                                # noqa: BLE001
            st.error("临时文件写入失败: %s" % exc)


# ---------------------------------------------------------------- 链选择
st.subheader("2) 指定链 (可留空自动推断)")

chain_map = {}
if pdb_path:
    try:
        from predict_affinity import _get_extractor
        chain_map = _get_extractor().list_protein_chains(pdb_path)
    except Exception as exc:                                    # noqa: BLE001
        st.warning("链解析失败: %s" % str(exc)[:90])

AUTO = "(自动推断最短链)"
c1, c2 = st.columns(2)
with c1:
    lig_sel = st.selectbox("配体链 (Ligand)", [AUTO] + sorted(chain_map.keys()))
with c2:
    rec_sel = st.selectbox("受体链 (Receptor, 可多选则用逗号)",
                           [AUTO] + sorted(chain_map.keys()))
lig_extra = st.text_input("或手动输入配体链 (优先级更高, 逗号分隔)", "")
rec_extra = st.text_input("或手动输入受体链 (优先级更高, 逗号分隔)", "")

if chain_map:
    st.caption("检测到蛋白链: %s" % ", ".join(
        "%s(%d残基)" % (k, v) for k, v in sorted(chain_map.items())))


# ---------------------------------------------------------------- 预测
st.subheader("3) 预测")

run = st.button("开始预测", type="primary")

if run:
    if not pdb_path:
        st.error("请先选择或上传一个结构文件。")
    else:
        lig = lig_extra.strip() or ("" if lig_sel == AUTO else lig_sel)
        rec = rec_extra.strip() or ("" if rec_sel == AUTO else rec_sel)
        try:
            with st.spinner("正在提取物理特征并预测 (首次约 3 秒)..."):
                r = get_prediction(pdb_path, lig, rec, MODEL_PATH)
        except Exception as exc:                                # noqa: BLE001
            st.error("预测失败: %s" % exc)
            with st.expander("详细错误"):
                st.code(traceback.format_exc())
            r = None
        finally:
            # 清理上传文件的临时副本 (不污染 data/pdb)
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:                               # noqa: BLE001
                    pass
                tmp_path = None

        if r:
            # ---- 大字号: 相对排名 ----
            if r["rank_pct"] is not None:
                st.markdown(
                    "### 相对排名: 强于训练集中的 "
                    "<span style='font-size:44px;color:#c0392b;font-weight:700'>"
                    "%.1f%%</span> 的界面" % r["rank_pct"],
                    unsafe_allow_html=True)
            else:
                st.warning("无法计算相对排名 (训练集统计缺失)")

            st.error("⚠️ 模型绝对数值存在收缩，请以相对排名为准")

            # ---- 指标 ----
            k1, k2, k3 = st.columns(3)
            k1.metric("预测 ΔG (raw)", "%+.2f kcal/mol" % r["raw_dG"])
            k2.metric("斜率校准 ΔG", "%+.2f kcal/mol" % r["calibrated_dG"],
                      help="按测试集拟合斜率还原; 定量更接近真实, 但噪声被放大")
            if r["ci_low"] is not None:
                k3.metric("95%% 经验区间".replace("%%", "%"),
                          "%+.2f ~ %+.2f" % (r["ci_low"], r["ci_high"]),
                          help="基于测试集 RMSE=%.2f 推导, 非严格后验区间" % r["rmse"])
            else:
                k3.metric("95% 经验区间", "不可用")

            st.caption(
                "配体链 %s | 受体链 %s | 界面接触对 %d | 氢键 %d | 盐桥 %d | ΔSASA %.0f Å²"
                % (",".join(r["ligand_chains"]), ",".join(r["receptor_chains"]),
                   r["n_contact_pairs"], r["n_hbond"], r["n_salt_bridge"],
                   r["dsasa_total"]))

            for w in r["warnings"]:
                st.warning(w)

            # ---- Top 5 物理特征柱状图 ----
            st.markdown("#### 该候选的物理特征 (Top 5, 按量级选取)")
            feats = r["top_features"]
            if feats:
                # 取量级最大的 5 个可读特征 (全 0/NaN 的不展示)
                ranked = sorted([f for f in feats if f[1] and abs(f[1]) > 1e-9],
                                key=lambda x: -abs(x[1]))[:5]
                if ranked:
                    df = pd.DataFrame(
                        {"物理特征": [x[0] for x in ranked],
                         "数值": [x[1] for x in ranked]}).set_index("物理特征")
                    st.bar_chart(df, height=300)
                    st.dataframe(df.T, width="stretch")
                else:
                    st.info("特征值全为 0, 无法绘图。")
            else:
                st.info("无可用特征。")

            with st.expander("全部指标与诊断"):
                st.json({k: v for k, v in r.items()
                         if k not in ("top_features", "chain_residues")})
                st.write("链残基数:", r["chain_residues"])
                if r["missing_features"]:
                    st.write("缺失特征 (已用训练中位数兜底):",
                             ", ".join(r["missing_features"]))

# 若脚本因异常提前结束, 兜底清理临时文件
if tmp_path and os.path.exists(tmp_path):
    try:
        os.remove(tmp_path)
    except Exception:                                           # noqa: BLE001
        pass

st.divider()
st.caption("仅供科研演示 | 模型排序能力强于定量能力 | 训练数据: PPB-Affinity (家族隔离划分)")

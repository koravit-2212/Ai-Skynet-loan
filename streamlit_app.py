import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="Loan Default Risk Demo", page_icon="🏦", layout="centered")

HERE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load():
    with open(os.path.join(HERE, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    booster = xgb.Booster()
    booster.load_model(os.path.join(HERE, "model.json"))
    return meta, booster


META, BOOSTER = load()
FEATS = META["feature_names"]
MED = pd.Series(META["medians"])
THR = META["threshold"]
FORM = META["form_features"]
M = META["metrics"]


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1%}"


def score(df):
    # feature ที่ไม่ได้กรอกจะถูกเติมด้วยค่ามัธยฐานของข้อมูลเทรน
    X = df.reindex(columns=FEATS).apply(pd.to_numeric, errors="coerce")
    X = X.fillna(MED).astype("float32")
    return BOOSTER.predict(xgb.DMatrix(X[FEATS]))


def build_metrics():
    return "\n".join([
        "## ผลการวัดโมเดลบนชุด test (ข้อมูลที่โมเดลไม่เคยเห็น)",
        "",
        f"- ชุดข้อมูล: **{META['dataset']}** (train {META['n_train']:,} แถว, test {META['n_test']:,} แถว)",
        f"- ROC-AUC: **{M['roc_auc']:.4f}** | PR-AUC: **{M['pr_auc']:.4f}** "
        f"(ค่าเทียบเคียงของ PR-AUC เมื่อเดาสุ่มคือสัดส่วนผู้ผิดนัด ≈ {META['positive_rate']:.1%})",
        f"- ที่เกณฑ์ {THR:.4f} (เลือกจากชุด validation ให้ recall ≥ {META['target_recall']:.0%}): "
        f"precision **{M['precision']:.1%}**, recall **{M['recall']:.1%}**, F1 **{M['f1']:.3f}**",
        f"- feature ที่สำคัญที่สุด: {', '.join(META['top_features'])}",
    ])


def build_rai():
    L = []
    L.append("## Responsible AI")
    L.append("")
    L.append("**วัตถุประสงค์:** เดโมเชิงการศึกษาเพื่อแสดงการทำนายความเสี่ยงผิดนัดชำระด้วยแมชชีนเลิร์นนิง "
             "**ไม่ใช่เครื่องมือตัดสินสินเชื่อจริง** ห้ามใช้อนุมัติหรือปฏิเสธเงินกู้โดยอัตโนมัติ")
    L.append("")
    L.append("**ข้อมูล:** LendingClub ปี 2007-2018 (CC0) มีข้อจำกัดสำคัญ")
    L.append("- มีเฉพาะผู้สมัครที่ **ได้รับอนุมัติ** แล้ว ผู้ที่ถูกปฏิเสธไม่อยู่ในข้อมูล โมเดลจึงเรียนจากกลุ่มที่ผ่านการคัดกรองมาก่อน (selection bias)")
    L.append("- ข้อมูลเก่า พฤติกรรมผู้กู้และเศรษฐกิจเปลี่ยนไปแล้ว ความแม่นยำกับผู้กู้ปัจจุบันอาจต่ำกว่าที่วัดได้ (data drift)")
    L.append("- แบ่งข้อมูลแบบสุ่ม ไม่ได้ทดสอบข้ามช่วงเวลา ผลจริงกับผู้กู้รุ่นใหม่อาจต่ำกว่านี้")
    L.append("- เป็นตลาดสหรัฐ ไม่ได้ตรวจกับผู้กู้ในประเทศอื่น")
    L.append("")
    L.append("**ผลการวัดและความหมาย:** " +
             f"ที่เกณฑ์ปัจจุบัน ทุก 100 รายที่โมเดลเตือนจะเป็นผู้ผิดนัดจริงประมาณ {M['precision']*100:.0f} ราย "
             f"และโมเดลจับผู้ผิดนัดจริงได้ {M['recall']:.0%} ที่เหลือพลาดไป ส่วนที่ถูกเตือนผิดอาจเสียโอกาส ถ้านำไปใช้จริงต้องมีคนตรวจซ้ำ")
    L.append("")
    L.append("**ข้อจำกัดของตัวเลขที่แสดง:**")
    L.append("- `risk score` ไม่ใช่ความน่าจะเป็นผิดนัดที่แท้จริง เพราะเทรนด้วยการถ่วงน้ำหนักคลาส ใช้เทียบกับเกณฑ์ได้ แต่อย่าอ่านเป็นเปอร์เซ็นต์")
    L.append("- ฟอร์มในแท็บ 'ลองทำนาย' กรอกเฉพาะ 10 feature ที่สำคัญสุด ที่เหลือถูกเติมด้วยค่ามัธยฐานของข้อมูลเทรน ผลจึงเป็นค่าประมาณ")
    L.append("- ค่าเกณฑ์เลือกให้จับผู้ผิดนัดได้มาก (recall สูง) แลกกับการเตือนผิดมากขึ้น ถ้าบริบทจริงให้น้ำหนักต่างกัน ต้องปรับเกณฑ์")
    L.append("")
    L.append("**ความเป็นธรรม:** ข้อมูลไม่มีเชื้อชาติหรือเพศเป็น feature โดยตรง แต่ feature อื่น (เช่น พื้นที่ รายได้) อาจเป็นตัวแทนของกลุ่มคุ้มครองได้ "
             "ยังไม่ได้ตรวจครบทุกกลุ่ม")
    fr = META.get("fairness")
    if fr:
        L.append("")
        L.append(f"ผลตรวจกลุ่มตาม `{META.get('fairness_col')}` บนชุด test (เชิงพรรณนา ความต่างอาจมาจากอัตราผิดนัดจริงที่ต่างกัน):")
        L.append("")
        L.append("| กลุ่ม | จำนวน | ผิดนัดจริง | ถูกเตือน | recall | precision |")
        L.append("|---|---|---|---|---|---|")
        for r in fr:
            L.append(f"| {r['group']} | {r['n']:,} | {pct(r['default_rate'])} | {pct(r['flag_rate'])} | "
                     f"{pct(r['recall'])} | {pct(r['precision'])} |")
    L.append("")
    L.append("**ความโปร่งใส:** feature ที่โมเดลให้น้ำหนักมากสุด: " + ", ".join(META["top_features"]) +
             ". ผู้ใช้ควรได้รับคำอธิบายและช่องทางโต้แย้งผลเสมอเมื่อมีการใช้โมเดลลักษณะนี้จริง")
    L.append("")
    L.append("**ความเป็นส่วนตัว:** อย่ากรอกข้อมูลส่วนบุคคลจริง โค้ดของแอปนี้ไม่บันทึกข้อมูลที่กรอกหรืออัปโหลด")
    L.append("")
    L.append("**การกำกับดูแลโดยมนุษย์:** ผล 'เกินเกณฑ์' หมายถึงควรให้คนตรวจสอบ ไม่ใช่ปฏิเสธอัตโนมัติ")
    return "\n".join(L)


st.title("Loan Default Risk Demo")
st.caption("เดโมทำนายความเสี่ยงผิดนัดชำระสินเชื่อ (เพื่อการศึกษา ไม่ใช่คำแนะนำทางการเงิน)")

tab1, tab2, tab3, tab4 = st.tabs(["ลองทำนาย", "อัปโหลด CSV", "ผลการวัดโมเดล", "Responsible AI"])

with tab1:
    st.write("กรอกค่าของผู้กู้สมมติ feature ที่ไม่ได้แสดงจะใช้ค่ามัธยฐานของข้อมูลเทรน")
    with st.form("one"):
        vals = [st.number_input(f["name"], value=float(round(f["median"], 4)),
                                help=f"ช่วงทั่วไป {f['min']:.4g} ถึง {f['max']:.4g}") for f in FORM]
        submitted = st.form_submit_button("ประเมินความเสี่ยง")
    if submitted:
        row = {f["name"]: v for f, v in zip(FORM, vals)}
        s = float(score(pd.DataFrame([row]))[0])
        c1, c2 = st.columns(2)
        c1.metric("risk score (0-1, ไม่ใช่ความน่าจะเป็นจริง)", f"{s:.4f}")
        c2.metric("เกณฑ์ (threshold)", f"{THR:.4f}")
        if s >= THR:
            st.warning("เกินเกณฑ์: ควรส่งให้เจ้าหน้าที่ตรวจสอบ")
        else:
            st.success("ต่ำกว่าเกณฑ์: ไม่ถูกเตือนโดยโมเดล")

with tab2:
    st.write("อัปโหลด CSV ที่มีคอลัมน์ชื่อเดียวกับ feature ของโมเดล (ไม่เกิน 5,000 แถว แสดงผล 200 แถวแรก)")
    up = st.file_uploader("CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up).head(5000)
        if not any(c in FEATS for c in df.columns):
            st.error("ไม่พบคอลัมน์ที่ตรงกับ feature ของโมเดลเลย ตรวจชื่อคอลัมน์ใน CSV")
        else:
            s = score(df)
            out = df.copy()
            out["risk_score"] = np.round(s, 4)
            out["flag_for_review"] = (s >= THR).astype(int)
            st.dataframe(out.head(200))
            st.download_button("ดาวน์โหลดผลลัพธ์ (CSV)", out.to_csv(index=False).encode("utf-8"),
                               file_name="risk_scores.csv", mime="text/csv")

with tab3:
    st.markdown(build_metrics())

with tab4:
    st.markdown(build_rai())

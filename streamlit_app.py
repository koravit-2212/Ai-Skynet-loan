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

# คำอธิบายตัวแปร (อิงชื่อคอลัมน์มาตรฐานของข้อมูล LendingClub)
# เพิ่มหรือแก้ได้โดยสร้างไฟล์ feature_descriptions.json ไว้ข้าง ๆ แอป รูปแบบ {"ชื่อคอลัมน์": "คำอธิบาย"}
GLOSSARY = {
    "loan_amnt": "ยอดเงินกู้ที่ผู้กู้ขอ (ดอลลาร์)",
    "funded_amnt": "ยอดเงินกู้ที่ได้รับอนุมัติจริง (ดอลลาร์)",
    "term": "ระยะเวลาผ่อนชำระ (เดือน) ปกติ 36 หรือ 60",
    "int_rate": "อัตราดอกเบี้ยของสินเชื่อ (% ต่อปี) ดอกเบี้ยสูงมักสะท้อนว่าผู้กู้เสี่ยงกว่า",
    "installment": "ค่างวดที่ต้องจ่ายต่อเดือน (ดอลลาร์)",
    "grade": "เกรดสินเชื่อที่ LendingClub ประเมิน (A ดีสุด ถึง G เสี่ยงสุด)",
    "sub_grade": "เกรดย่อยของสินเชื่อ (เช่น A1-G5) ละเอียดกว่า grade",
    "emp_length": "อายุงานของผู้กู้ (ปี)",
    "annual_inc": "รายได้ต่อปีของผู้กู้ที่แจ้งไว้ (ดอลลาร์)",
    "dti": "ภาระหนี้ต่อรายได้ (%) = หนี้ที่ต้องจ่ายต่อเดือนหารด้วยรายได้ต่อเดือน (ไม่รวมสินเชื่อบ้านและสินเชื่อนี้) ยิ่งสูงยิ่งเสี่ยง",
    "fico_range_low": "คะแนนเครดิต FICO ช่วงล่าง ณ ตอนขอกู้",
    "fico_range_high": "คะแนนเครดิต FICO ช่วงบน ณ ตอนขอกู้",
    "fico_score": "คะแนนเครดิต FICO ณ ตอนขอกู้ (ยิ่งสูงยิ่งดี)",
    "delinq_2yrs": "จำนวนครั้งที่ค้างชำระเกิน 30 วันในช่วง 2 ปีที่ผ่านมา",
    "inq_last_6mths": "จำนวนครั้งที่ถูกสอบถามเครดิตในช่วง 6 เดือนที่ผ่านมา",
    "mths_since_last_delinq": "จำนวนเดือนนับจากการค้างชำระครั้งล่าสุด",
    "mths_since_last_record": "จำนวนเดือนนับจากบันทึกสาธารณะเชิงลบครั้งล่าสุด",
    "open_acc": "จำนวนบัญชีเครดิตที่ยังเปิดอยู่",
    "total_acc": "จำนวนบัญชีเครดิตทั้งหมดที่เคยมี",
    "pub_rec": "จำนวนบันทึกสาธารณะเชิงลบ (เช่น คดีหนี้ล้มละลาย)",
    "pub_rec_bankruptcies": "จำนวนบันทึกการล้มละลายสาธารณะ",
    "revol_bal": "ยอดหนี้หมุนเวียนคงค้าง เช่น บัตรเครดิต (ดอลลาร์)",
    "revol_util": "สัดส่วนการใช้วงเงินหมุนเวียน (%) ยิ่งใกล้ 100 ยิ่งใช้วงเงินเต็ม",
    "mort_acc": "จำนวนบัญชีสินเชื่อบ้าน",
    "avg_cur_bal": "ยอดคงค้างเฉลี่ยต่อบัญชีที่ยังเปิด (ดอลลาร์)",
    "bc_util": "สัดส่วนการใช้วงเงินบัตรเครดิต (%)",
    "tot_cur_bal": "ยอดหนี้คงค้างรวมทุกบัญชี (ดอลลาร์)",
    "tot_hi_cred_lim": "วงเงินสินเชื่อสูงสุดรวม (ดอลลาร์)",
    "total_bc_limit": "วงเงินบัตรเครดิตรวม (ดอลลาร์)",
    "total_rev_hi_lim": "วงเงินหมุนเวียนสูงสุดรวม (ดอลลาร์)",
    "num_actv_bc_tl": "จำนวนบัญชีบัตรเครดิตที่ยังใช้งาน",
    "num_bc_sats": "จำนวนบัญชีบัตรเครดิตที่สถานะดี",
    "num_rev_accts": "จำนวนบัญชีหมุนเวียนทั้งหมด",
    "num_il_tl": "จำนวนบัญชีผ่อนชำระเป็นงวด",
    "percent_bc_gt_75": "สัดส่วนบัญชีบัตรเครดิตที่ใช้วงเงินเกิน 75% (%)",
    "acc_open_past_24mths": "จำนวนบัญชีที่เปิดใหม่ใน 24 เดือนที่ผ่านมา",
    "mo_sin_old_rev_tl_op": "อายุเป็นเดือนของบัญชีหมุนเวียนที่เก่าที่สุด",
    "mo_sin_rcnt_tl": "จำนวนเดือนนับจากเปิดบัญชีล่าสุด",
    "tax_liens": "จำนวนบันทึกภาษีค้างชำระ (tax lien)",
    "earliest_cr_line": "ปีหรือวันที่เปิดบัญชีเครดิตแรก (ประวัติเครดิตยาวนานแค่ไหน)",
    "issue_year": "ปีที่ออกสินเชื่อ",
    "addr_state": "รัฐที่ผู้กู้อยู่ (รหัสตัวเลขที่แปลงมา)",
    "home_ownership": "สถานะที่อยู่อาศัย เช่น เช่า จำนอง เจ้าของ (รหัสตัวเลขที่แปลงมา)",
    "purpose": "วัตถุประสงค์การกู้ เช่น รวมหนี้ ซ่อมบ้าน (รหัสตัวเลขที่แปลงมา)",
    "verification_status": "สถานะการตรวจสอบรายได้โดย LendingClub (รหัสตัวเลขที่แปลงมา)",
}


import re

# แปลคำศัพท์ในชื่อคอลัมน์เป็นไทย ใช้สร้างคำอธิบายโดยประมาณเมื่อชื่อไม่อยู่ใน GLOSSARY
TOKENS = {
    "loan": "เงินกู้", "amnt": "ยอด", "amount": "ยอด", "funded": "ที่ได้รับอนุมัติ", "int": "ดอกเบี้ย",
    "rate": "อัตรา", "installment": "ค่างวด", "term": "ระยะเวลาผ่อน", "grade": "เกรด", "sub": "ย่อย",
    "emp": "การจ้างงาน", "length": "ระยะเวลา", "annual": "ต่อปี", "inc": "รายได้", "income": "รายได้",
    "monthly": "ต่อเดือน", "dti": "ภาระหนี้ต่อรายได้", "fico": "คะแนนเครดิต FICO", "score": "คะแนน",
    "range": "ช่วง", "low": "ต่ำ", "high": "สูง", "delinq": "การค้างชำระ", "inq": "การสอบถามเครดิต",
    "pub": "สาธารณะ", "rec": "บันทึก", "record": "บันทึก", "bankruptcies": "การล้มละลาย", "tax": "ภาษี",
    "liens": "การอายัดทรัพย์", "collections": "การเรียกเก็บหนี้", "chargeoff": "การตัดหนี้สูญ",
    "open": "ที่เปิดอยู่", "op": "ที่เปิดอยู่", "acc": "บัญชี", "accts": "บัญชี", "acct": "บัญชี",
    "tl": "บัญชีเครดิต", "bc": "บัตรเครดิต", "il": "สินเชื่อผ่อนงวด", "rev": "หมุนเวียน",
    "revol": "หมุนเวียน", "mort": "สินเชื่อบ้าน", "mortgage": "สินเชื่อบ้าน", "sats": "ที่สถานะดี",
    "actv": "ที่ยังใช้งาน", "num": "จำนวน", "tot": "รวม", "total": "รวม", "avg": "เฉลี่ย", "max": "สูงสุด",
    "min": "ต่ำสุด", "pct": "สัดส่วน (%)", "percent": "สัดส่วน (%)", "ratio": "อัตราส่วน",
    "util": "การใช้วงเงิน", "bal": "ยอดคงค้าง", "cur": "ปัจจุบัน", "lim": "วงเงิน", "limit": "วงเงิน",
    "cred": "เครดิต", "credit": "เครดิต", "hi": "สูงสุด", "debt": "หนี้", "mths": "เดือน", "mth": "เดือน",
    "mo": "เดือน", "months": "เดือน", "yrs": "ปี", "yr": "ปี", "years": "ปี", "year": "ปี",
    "since": "นับจาก", "sin": "นับจาก", "last": "ล่าสุด", "rcnt": "ล่าสุด", "recent": "ล่าสุด",
    "old": "เก่าที่สุด", "oldest": "เก่าที่สุด", "earliest": "แรกสุด", "past": "ที่ผ่านมา",
    "issue": "ที่ออกสินเชื่อ", "cr": "เครดิต", "line": "บัญชี", "gt": "มากกว่า", "lt": "น้อยกว่า",
    "addr": "ที่อยู่", "state": "รัฐ", "home": "ที่อยู่อาศัย", "ownership": "การถือครอง",
    "purpose": "วัตถุประสงค์", "verification": "การตรวจสอบ", "status": "สถานะ", "zip": "รหัสไปรษณีย์",
    "missing": "ค่าที่ขาดหาย", "flag": "ตัวบ่งชี้", "is": "ตัวบ่งชี้ว่า", "has": "ตัวบ่งชี้ว่ามี",
    "new": "ใหม่", "all": "ทั้งหมด", "pymnt": "การชำระเงิน", "payment": "การชำระเงิน", "type": "ประเภท",
    "joint": "ร่วม", "app": "ใบสมัคร",
}


def auto_describe(name):
    parts = [p for p in name.lower().replace("-", "_").split("_") if p]
    pre = ""
    if parts and parts[0] in ("log", "sqrt"):
        pre = "ค่าลอการิทึมของ " if parts[0] == "log" else "รากที่สองของ "
        parts = parts[1:]
    out, known = [], 0
    for p in parts:
        m = re.fullmatch(r"(\d+)([a-z]+)", p)
        if p in TOKENS:
            out.append(TOKENS[p]); known += 1
        elif m and m.group(2) in TOKENS:
            out.append(m.group(1) + " " + TOKENS[m.group(2)]); known += 1
        else:
            out.append(p)
    if known == 0:
        return f"ค่าตัวเลขของตัวแปร '{name}' (ยังไม่มีคำแปลสำหรับชื่อนี้ ดูความหมายจาก data dictionary ของชุดข้อมูล)"
    return f"{pre}{' '.join(out)} (แปลจากชื่อตัวแปร '{name}' โดยประมาณ)"


def load_descriptions():
    d = dict(GLOSSARY)
    p = os.path.join(HERE, "feature_descriptions.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            d.update(json.load(f))
    return d


DESC = load_descriptions()


def describe(name):
    return DESC[name] if name in DESC else auto_describe(name)


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
    with st.expander("อ่านผลลัพธ์อย่างไร"):
        st.markdown(
            "- **risk score**: คะแนนความเสี่ยงผิดนัด 0-1 ยิ่งสูงยิ่งเสี่ยง "
            "**ไม่ใช่ความน่าจะเป็นจริง** ใช้เทียบกับเกณฑ์เท่านั้น\n"
            "- **เกณฑ์ (threshold)**: ถ้า risk score ถึงเกณฑ์นี้ โมเดลจะเตือน "
            "เลือกจากชุด validation ให้จับผู้ผิดนัดได้ตามเป้า (recall)\n"
            "- **ผลการประเมิน**: 'เกินเกณฑ์' แปลว่าควรให้เจ้าหน้าที่ตรวจสอบ ไม่ใช่การปฏิเสธอัตโนมัติ")
    with st.form("one"):
        vals = []
        for f in FORM:
            vals.append(st.number_input(f["name"], value=float(round(f["median"], 4))))
            st.caption(f"{describe(f['name'])} | ช่วงทั่วไป {f['min']:.4g} ถึง {f['max']:.4g} "
                       f"(ค่าตั้งต้นคือค่ากลางของข้อมูล)")
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

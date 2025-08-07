# food_info_chat_app.py
"""
🍉 Info Bahan Pangan – Chat-Based App (YOLOv8 + Llama-3)
=======================================================

Fitur singkat
-------------
• Upload foto ➜ deteksi bahan ➜ tabel info + ringkasan.  
• Chat aktif; memori 10 pesan.  
• Spinner “Menyiapkan jawaban…” muncul saat LLM bekerja.
"""
from __future__ import annotations
import os, re, tempfile
from pathlib import Path
from typing import List, Set

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ————— CONFIG —————
PAGE_TITLE = "🍉 Info Bahan Pangan – Chat"
PAGE_ICON  = "🍉"
MODEL_PATH = Path("best.pt")
DATA_PATH  = Path("dataset_info_bahan_pangan_mentah.csv")
MEMORY_WINDOW = 10

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

# LLM (Groq – Llama-3)
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# ——— Cached loaders ———
@st.cache_resource(show_spinner=False)
def load_yolo(path: Path) -> YOLO:
    if not path.exists():
        st.error("❌ Model YOLO tidak ditemukan."); st.stop()
    return YOLO(str(path))

@st.cache_resource(show_spinner=False)
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error("❌ Dataset tidak ditemukan."); st.stop()
    df = pd.read_csv(path)
    required = {
        "nama","kalori_100g","usia_min_bulan","alergen","kelompok_rentan",
        "sumber","saran_konsumsi_sehat","konsumsi_tidak_sehat",
    }
    if not required.issubset(df.columns):
        st.error("❌ Kolom dataset belum sesuai skema."); st.stop()
    df["nama_lower"] = df["nama"].str.lower().str.strip()
    return df

# ——— Helper functions ———
def detect_objects(path:str, model:YOLO, conf:float=0.25):
    res = model.predict(source=path, device="cpu", save=False,
                        conf=conf, verbose=False)
    foods: List[str] = []
    for r in res:
        if r.boxes is not None:
            ids = r.boxes.cls.cpu().numpy().astype(int)
            foods.extend(model.names.get(i, f"class_{i}") for i in ids)
    return sorted(set(foods)), res[0].plot()

def lookup_rows(names:Set[str], df:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for n in names:
        m=df[df["nama_lower"]==n.lower().strip()]
        if not m.empty:
            rows.append(m.iloc[0][[
                "nama","kalori_100g","usia_min_bulan","alergen",
                "kelompok_rentan","saran_konsumsi_sehat",
                "konsumsi_tidak_sehat","sumber",
            ]])
    return pd.DataFrame(rows).reset_index(drop=True)

def extract_ingredients(txt:str, df:pd.DataFrame)->Set[str]:
    low = txt.lower()
    return {nm for nm in df["nama_lower"].unique()
            if re.search(rf"\b{re.escape(nm)}\b", low)}

def history_to_text(hist:List[dict])->str:
    lines=[]
    for h in hist[-MEMORY_WINDOW:]:
        who = "Pengguna" if h["role"]=="user" else "Ahli gizi"
        lines.append(f"{who}: {h['content']}")
    return "\n".join(lines)

# ——— Prompts ———
SUMMARY_PROMPT = """
Kamu adalah ahli gizi. Berdasarkan tabel (kolom: nama, kalori_100g, usia_min_bulan,
alergen, kelompok_rentan, saran_konsumsi_sehat, konsumsi_tidak_sehat, sumber):
{table_md}

Tuliskan satu paragraf ringkas (≤150 kata) merangkum kalori, batas usia,
risiko alergen/kelompok rentan, dan saran konsumsi sehat vs tidak sehat.
Gunakan hanya data tabel.
"""
FOLLOW_UP_PROMPT = """
Kamu adalah ahli gizi digital yang hanya boleh menjawab berdasarkan data dalam tabel di bawah ini.

TABEL DATA NUTRISI (wajib dijadikan acuan):
{table_md}

Riwayat percakapan terakhir:
{history}

Pertanyaan terbaru dari pengguna:
"{question}"

INSTRUKSI:
- Jawaban harus SINGKAT (maksimum 120 kata).
- Jangan menyebut bahan makanan lain yang tidak ada di tabel.
- Jangan mengarang info tambahan (seperti ikan, daging olahan, kulit/lemak) jika tidak ada dalam tabel.
- Jika bahan yang ditanya tidak ada dalam tabel, cukup jawab: "Maaf, bahan tersebut tidak tersedia dalam data kami."
- Fokus hanya pada kolom: kalori_100g, usia_min_bulan, alergen, kelompok_rentan, saran_konsumsi_sehat, konsumsi_tidak_sehat.

Berikan jawaban dalam Bahasa Indonesia yang jelas dan tidak mengada-ada.
"""


def llm_call(prompt:str)->str:
    if ChatGroq is None or not GROQ_API_KEY:
        return "(LLM tidak tersedia—atur GROQ_API_KEY & instal langchain_groq.)"
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
    return getattr(llm.invoke(prompt), "content", "").strip()

# ——— App state ———
model     = load_yolo(MODEL_PATH)
master_df = load_dataset(DATA_PATH)

if "messages" not in st.session_state: st.session_state.messages=[]
if "ready"    not in st.session_state: st.session_state.ready=False
if "context_set" not in st.session_state: st.session_state.context_set:set[str]=set()
if "context_df"  not in st.session_state: st.session_state.context_df=pd.DataFrame()

# ——— Sidebar ———
with st.sidebar:
    st.header("📸 Unggah foto bahan panganmu & temukan info gizinya seketika!")
    upload = st.file_uploader("Foto bahan pangan", type=["jpg","jpeg","png"],
                              disabled=st.session_state.ready)
    st.markdown(
        """
        ---
        **Disclaimer**  
        Info yang disajikan mencakup kalori, alergen, dan saran umum—bukan nasihat medis.  
        """,
        unsafe_allow_html=True
    )

    if upload and not st.session_state.ready:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            Image.open(upload).convert("RGB").save(tmp.name)
            tmp_path = tmp.name

        with st.spinner("Mendeteksi bahan…"):
            names, plot_img = detect_objects(tmp_path, model)
        os.remove(tmp_path)

        st.session_state.ready = True
        st.session_state.context_set.update(names)
        st.session_state.context_df = lookup_rows(set(names), master_df)
        st.success("✅ Foto berhasil diproses! Silakan ajukan pertanyaan.")

        st.session_state.messages.append(
            {"role":"user","content":"[Gambar di-unggah]", "image": upload}
        )
        summary = llm_call(
            SUMMARY_PROMPT.format(
                table_md=st.session_state.context_df.to_markdown(index=False))
        )
        st.session_state.messages.append(
            {"role":"assistant","content":summary,"image":plot_img,
             "table_df":st.session_state.context_df}
        )

# ——— Chat log ———
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if "image" in m: st.image(m["image"], use_container_width=True)
        st.markdown(m["content"])
        if m.get("table_df") is not None:
            st.dataframe(m["table_df"], hide_index=True, use_container_width=True)

# ——— Chat input + typing indicator ———
if st.session_state.ready:
    question = st.chat_input("Tanya atau sebut bahan tambahan…")
    if question:
        with st.chat_message("user"): st.markdown(question)
        st.session_state.messages.append({"role":"user","content":question})

        new_ing = extract_ingredients(question, master_df) - st.session_state.context_set
        if new_ing:
            extra = lookup_rows(new_ing, master_df)
            st.session_state.context_df = pd.concat(
                [st.session_state.context_df, extra], ignore_index=True)
            st.session_state.context_set.update(new_ing)

        mem = history_to_text(st.session_state.messages)

        with st.chat_message("assistant"):
            placeholder = st.empty()                     # tempat jawaban nanti
            placeholder.markdown("⏳ _Menyiapkan jawaban..._")
            ans = llm_call(
                FOLLOW_UP_PROMPT.format(
                    table_md=st.session_state.context_df.to_markdown(index=False),
                    history=mem, question=question
                )
            )
            placeholder.markdown(ans)                   # ganti teks spinner

        st.session_state.messages.append({"role":"assistant","content":ans})
else:
    st.chat_input("Silakan upload gambar terlebih dahulu…", disabled=True)

# ——— Footer ———
st.markdown(
    """
    <hr style='margin-top:1.5rem;margin-bottom:0.5rem'>
    <small><i>Disclaimer: informasi terbatas pada kalori, alergi, dan rekomendasi umum;
    bukan pengganti saran medis individual.</i></small>
    """,
    unsafe_allow_html=True
)
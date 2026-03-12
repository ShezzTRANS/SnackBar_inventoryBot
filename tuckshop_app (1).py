"""
Tuck Shop Inventory Monitor Chatbot
Pipeline: WhatsApp text → Excel update → ML prediction → HuggingFace chat
"""

import streamlit as st
from huggingface_hub import InferenceClient
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import openpyxl
from openpyxl import load_workbook

import re, json, datetime, io

st.set_page_config(page_title="Tuck Shop Monitor", page_icon="🏪", layout="wide")

# ── HuggingFace Auth ─────────────────────────────────────────────
try:
    hf_token = st.secrets["HUGGINGFACE_TOKEN"]
    client   = InferenceClient(token=hf_token)
except Exception:
    st.error("⚠️ Add HUGGINGFACE_TOKEN to Streamlit secrets.")
    st.stop()

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;background:#0f1a0f;color:#e8f5e9}
.stApp{background:#0f1a0f}
section[data-testid="stSidebar"]{background:#0a120a;border-right:1px solid #1b4332}
section[data-testid="stSidebar"] *{color:#a5d6a7!important}
.metric-card{background:#1b4332;border:1px solid #2d6a4f;border-radius:10px;padding:14px 18px;margin:6px 0}
.metric-card .val{font-size:1.6rem;font-weight:700;color:#69f0ae;font-family:'IBM Plex Mono',monospace}
.metric-card .lbl{font-size:.75rem;color:#81c784;text-transform:uppercase;letter-spacing:.08em}
.msg-user{background:#1b4332;border:1px solid #2d6a4f;border-radius:12px 12px 4px 12px;padding:10px 14px;margin:8px 0 8px 18%}
.msg-assistant{background:#0a1f0a;border:1px solid #1b4332;border-left:3px solid #69f0ae;border-radius:12px 12px 12px 4px;padding:10px 14px;margin:8px 18% 8px 0}
.msg-label{font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:#4a7c59;margin-bottom:3px;text-transform:uppercase;letter-spacing:.08em}
.stTextInput>div>div>input,.stTextArea textarea{background:#1b4332!important;color:#e8f5e9!important;border:1px solid #2d6a4f!important;border-radius:8px!important}
.stButton>button{background:#69f0ae!important;color:#0a1f0a!important;border:none!important;border-radius:6px!important;font-family:'IBM Plex Mono',monospace!important;font-weight:700!important}
.stButton>button:hover{background:#40c977!important}
.whatsapp-box{background:#1a2e1a;border:1px solid #2d5a2d;border-radius:10px;padding:14px;font-family:'IBM Plex Mono',monospace;font-size:.82rem;color:#a5d6a7;margin:8px 0}
hr{border-color:#1b4332!important}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# EXCEL HELPERS
# ════════════════════════════════════════════════════════════════

def load_excel(file_bytes: bytes) -> dict:
    """Returns {inventory: df, sales_log: df}"""
    xl = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    result = {}
    for name, df in xl.items():
        result[name.lower().replace(" ", "_")] = df
    return result

def parse_whatsapp_sales(text: str, inventory_df: pd.DataFrame) -> dict[str, int]:
    """
    Parses WhatsApp sales text like:
      'Sold: Coke x5, Chips x3, Bread x2'
      'coke 5, simba 3, bread 2'
    Returns {product_keyword: qty}
    """
    sales = {}
    # Match patterns: "word x5" or "word 5" or "word: 5"
    patterns = re.findall(r"([a-zA-Z]+(?:\s[a-zA-Z]+)?)\s*[x:×]?\s*(\d+)", text, re.IGNORECASE)
    for keyword, qty in patterns:
        keyword = keyword.strip().lower()
        qty = int(qty)
        # Fuzzy-match against product names
        if inventory_df is not None and "Product" in inventory_df.columns:
            for product in inventory_df["Product"].dropna():
                if keyword in product.lower() or product.lower().split()[0] in keyword:
                    sales[product] = sales.get(product, 0) + qty
                    break
        else:
            sales[keyword] = qty
    return sales


# ════════════════════════════════════════════════════════════════
# ML — LINEAR REGRESSION PROFIT PREDICTOR
# ════════════════════════════════════════════════════════════════

def run_ml_prediction(sales_log_df: pd.DataFrame) -> dict:
    """
    Trains a Linear Regression on historical daily profit.
    Returns predictions for next 3 days + model stats.
    """
    try:
        # Normalise column names
        df = sales_log_df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        # Find profit and revenue columns
        profit_col  = next((c for c in df.columns if "profit" in c.lower()), None)
        revenue_col = next((c for c in df.columns if "revenue" in c.lower()), None)

        if profit_col is None or len(df.dropna(subset=[profit_col])) < 3:
            return {"error": "Need at least 3 days of sales data for predictions."}

        df = df.dropna(subset=[profit_col])
        df = df.reset_index(drop=True)
        df["day_index"] = range(len(df))

        X = df[["day_index"]].values
        y_profit  = df[profit_col].astype(float).values
        y_revenue = df[revenue_col].astype(float).values if revenue_col else y_profit

        model_p = LinearRegression().fit(X, y_profit)
        model_r = LinearRegression().fit(X, y_revenue)

        # R² score
        r2 = model_p.score(X, y_profit)

        # Predict next 3 days
        n = len(df)
        future_X = np.array([[n], [n+1], [n+2]])
        pred_profit  = model_p.predict(future_X)
        pred_revenue = model_r.predict(future_X)

        today = datetime.date.today()
        predictions = []
        for i in range(3):
            date = today + datetime.timedelta(days=i+1)
            predictions.append({
                "date": str(date),
                "predicted_revenue": round(float(pred_revenue[i]), 2),
                "predicted_profit":  round(float(pred_profit[i]), 2),
            })

        # Trend
        slope = float(model_p.coef_[0])
        trend = "📈 Growing" if slope > 2 else ("📉 Declining" if slope < -2 else "➡️ Stable")

        return {
            "predictions": predictions,
            "r2_score": round(r2, 3),
            "trend": trend,
            "avg_daily_profit":  round(float(np.mean(y_profit)), 2),
            "avg_daily_revenue": round(float(np.mean(y_revenue)), 2),
            "best_day_profit":   round(float(np.max(y_profit)), 2),
        }
    except Exception as e:
        return {"error": str(e)}


# ════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — Inventory-aware
# ════════════════════════════════════════════════════════════════

def build_system_prompt(inventory_summary: str, ml_summary: str) -> str:
    return f"""You are an intelligent inventory and profit assistant for a small South African tuck shop.

You help the owner:
1. Understand their current stock levels (storeroom, floor, balance)
2. Calculate daily profit from WhatsApp sales messages
3. Interpret ML profit predictions
4. Identify which products are most profitable
5. Warn when stock is running low

CURRENT INVENTORY SNAPSHOT:
{inventory_summary}

ML PREDICTION SUMMARY:
{ml_summary}

Rules:
- Always use South African Rand (R) for prices
- Be concise and practical — owner is busy
- If asked to parse a WhatsApp message, extract product names and quantities
- If stock of any item falls below 5 units, flag it as LOW STOCK ⚠️
- Speak plainly, like a helpful business advisor"""


# ════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════

defaults = {
    "messages": [],
    "inventory_bytes": None,
    "inventory_data": None,
    "ml_results": None,
    "parsed_sales": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🏪 Tuck Shop Monitor")
    st.markdown("---")

    # Model selector
    MODELS = {
        "Qwen 2.5 72B ⭐": "Qwen/Qwen2.5-72B-Instruct",
        "Qwen 2.5 7B ⚡":  "Qwen/Qwen2.5-7B-Instruct",
        "Mistral Nemo 12B": "mistralai/Mistral-Nemo-Instruct-2407",
        "Zephyr 7B Beta":   "HuggingFaceH4/zephyr-7b-beta",
    }
    model_label  = st.selectbox("Model", list(MODELS.keys()))
    model_choice = MODELS[model_label]

    max_tokens  = st.slider("Max tokens", 100, 1024, 512)
    temperature = st.slider("Temperature", 0.1, 1.2, 0.4)

    st.markdown("---")
    st.markdown("**📂 Upload stock.xlsx**")
    uploaded = st.file_uploader("", type=["xlsx"], label_visibility="collapsed")

    if uploaded:
        st.session_state.inventory_bytes = uploaded.read()
        st.session_state.inventory_data  = load_excel(st.session_state.inventory_bytes)

        # Run ML on upload
        log_key = next((k for k in st.session_state.inventory_data
                        if "log" in k or "sales" in k), None)
        if log_key:
            st.session_state.ml_results = run_ml_prediction(
                st.session_state.inventory_data[log_key])
        st.success("✅ Stock file loaded!")

    st.markdown("---")
    st.markdown("**📱 Paste WhatsApp Sales Text**")
    wa_text = st.text_area("", placeholder="Sold: Coke x5, Chips x3, Bread x2",
                           label_visibility="collapsed", height=80)
    if st.button("📥 Parse Sales", use_container_width=True) and wa_text.strip():
        inv_key = next((k for k in (st.session_state.inventory_data or {})
                        if "invent" in k), None)
        inv_df  = st.session_state.inventory_data[inv_key] if inv_key else None
        st.session_state.parsed_sales = parse_whatsapp_sales(wa_text, inv_df)

        if st.session_state.parsed_sales:
            st.success("Parsed sales:")
            for p, q in st.session_state.parsed_sales.items():
                st.write(f"  • {p}: **{q} units**")
        else:
            st.warning("No products matched. Try: 'Coke x5, Chips x3'")

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ════════════════════════════════════════════════════════════════
# MAIN AREA — Dashboard + Chat
# ════════════════════════════════════════════════════════════════

st.markdown('<h1 style="color:#69f0ae;font-family:IBM Plex Mono,monospace;font-size:1.5rem">🏪 TUCK SHOP INVENTORY MONITOR</h1>', unsafe_allow_html=True)

# ── KPI cards ────────────────────────────────────────────────────
ml = st.session_state.ml_results
col1, col2, col3, col4 = st.columns(4)

with col1:
    val = f"R{ml['avg_daily_profit']:,.2f}" if ml and "avg_daily_profit" in ml else "—"
    st.markdown(f'<div class="metric-card"><div class="lbl">Avg Daily Profit</div><div class="val">{val}</div></div>', unsafe_allow_html=True)

with col2:
    val = f"R{ml['predictions'][0]['predicted_profit']:,.2f}" if ml and "predictions" in ml else "—"
    st.markdown(f'<div class="metric-card"><div class="lbl">Tomorrow\'s Prediction</div><div class="val">{val}</div></div>', unsafe_allow_html=True)

with col3:
    val = ml.get("trend", "—") if ml else "—"
    st.markdown(f'<div class="metric-card"><div class="lbl">Profit Trend</div><div class="val" style="font-size:1.1rem">{val}</div></div>', unsafe_allow_html=True)

with col4:
    val = f"R{ml['best_day_profit']:,.2f}" if ml and "best_day_profit" in ml else "—"
    st.markdown(f'<div class="metric-card"><div class="lbl">Best Day Profit</div><div class="val">{val}</div></div>', unsafe_allow_html=True)

# ── Parsed WhatsApp sales preview ─────────────────────────────
if st.session_state.parsed_sales:
    st.markdown("**📱 Latest WhatsApp Sales Parsed:**")
    cols = st.columns(len(st.session_state.parsed_sales))
    for col, (prod, qty) in zip(cols, st.session_state.parsed_sales.items()):
        col.markdown(f'<div class="metric-card"><div class="lbl">{prod}</div><div class="val">{qty} units</div></div>', unsafe_allow_html=True)

# ── 3-day prediction table ─────────────────────────────────────
if ml and "predictions" in ml:
    st.markdown("**🤖 ML 3-Day Forecast (Linear Regression)**")
    pred_df = pd.DataFrame(ml["predictions"])
    pred_df.columns = ["Date", "Predicted Revenue (R)", "Predicted Profit (R)"]
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    st.caption(f"Model R² score: {ml.get('r2_score', '?')} · Accuracy improves with more daily data")

st.markdown("---")

# ── Chat area ─────────────────────────────────────────────────
st.markdown("**💬 Ask your inventory assistant**")

for msg in st.session_state.messages:
    css = "msg-user" if msg["role"] == "user" else "msg-assistant"
    st.markdown(
        f'<div class="{css}"><div class="msg-label">{msg["role"]}</div>{msg["content"]}</div>',
        unsafe_allow_html=True)

inp_col, btn_col = st.columns([8, 1])
with inp_col:
    user_input = st.text_input("", placeholder='e.g. "What\'s my profit today?" or paste WhatsApp sales text',
                                label_visibility="collapsed", key="chat_input")
with btn_col:
    send = st.button("Send", use_container_width=True)

# Suggested prompts
st.markdown(
    '<div class="whatsapp-box">💡 Try: &nbsp;"What is my profit today?" &nbsp;|&nbsp; '
    '"Which product makes the most profit?" &nbsp;|&nbsp; '
    '"Sold: Coke x5, Chips x3, Bread x2 — update my stock" &nbsp;|&nbsp; '
    '"What will I earn tomorrow?"</div>',
    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# SEND HANDLER
# ════════════════════════════════════════════════════════════════

if send and user_input.strip():

    # Build context summaries for system prompt
    inv_summary = "No stock file uploaded yet."
    if st.session_state.inventory_data:
        inv_key = next((k for k in st.session_state.inventory_data if "invent" in k), None)
        if inv_key:
            df = st.session_state.inventory_data[inv_key].dropna(how="all")
            inv_summary = df.to_string(index=False, max_rows=15)

    ml_summary = "No ML results yet — upload stock.xlsx with sales history."
    if ml and "predictions" in ml:
        preds = ml["predictions"]
        ml_summary = (
            f"Trend: {ml['trend']} | Avg daily profit: R{ml['avg_daily_profit']} | "
            f"Tomorrow forecast: R{preds[0]['predicted_profit']} profit, R{preds[0]['predicted_revenue']} revenue | "
            f"R² accuracy: {ml['r2_score']}"
        )

    # Include parsed WhatsApp sales in message if present
    full_user = user_input.strip()
    if st.session_state.parsed_sales:
        sales_str = ", ".join(f"{p}: {q}" for p, q in st.session_state.parsed_sales.items())
        full_user += f"\n\n[Parsed WhatsApp sales: {sales_str}]"

    st.session_state.messages.append({"role": "user", "content": full_user})

    system_prompt = build_system_prompt(inv_summary, ml_summary)

    messages_for_api = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    placeholder = st.empty()

    try:
        stream = client.chat_completion(
            model=model_choice,
            messages=messages_for_api,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        reply = ""
        for chunk in stream:
            token = (chunk.choices[0].delta.content
                     if chunk.choices and chunk.choices[0].delta.content else "")
            reply += token
            placeholder.markdown(
                f'<div class="msg-assistant"><div class="msg-label">assistant</div>{reply}▌</div>',
                unsafe_allow_html=True)

        placeholder.empty()
        st.session_state.messages.append({"role": "assistant", "content": reply})

    except Exception as e:
        st.error(f"⚠️ {e}")

    st.rerun()

"""Streamlit UI — Retail Sales Forecasting with POS upload + auth."""

import streamlit as st
import requests
import pandas as pd
from datetime import date

# ---------- PAGE CONFIG MUST BE FIRST ----------
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# ============================================================
# SIDEBAR (rendered early so it always shows)
# ============================================================
st.sidebar.title("🛒 Retail Forecast")
st.sidebar.caption(f"API: `{API_URL}`")

# API key input
default_key = st.session_state.get("api_key", "admin-key-change-me")
api_key = st.sidebar.text_input(
    "🔑 API Key",
    value=default_key,
    type="password",
    help="admin-key-change-me | analyst-key-change-me | viewer-key-change-me",
)
st.session_state["api_key"] = api_key
HEADERS = {"X-API-Key": api_key} if api_key else {}

# Connection status
USER_ROLE = None
try:
    r = requests.get(f"{API_URL}/health", timeout=3)
    if r.status_code == 200:
        health = r.json()
        st.sidebar.success("✅ API online")
        st.sidebar.caption(
            f"Model {'✅' if health.get('model_loaded') else '❌'}  •  "
            f"History {'✅' if health.get('history_loaded') else '❌'}"
        )
    else:
        st.sidebar.error(f"⚠️ API returned {r.status_code}")
except Exception as e:
    st.sidebar.error("❌ API unreachable")
    st.sidebar.caption(str(e))

# User info
try:
    me = requests.get(f"{API_URL}/auth/whoami", headers=HEADERS, timeout=3)
    if me.status_code == 200:
        user = me.json()
        st.sidebar.info(f"👤 **{user['user']}**  ({user['role']})")
        USER_ROLE = user["role"]
    else:
        st.sidebar.warning("🔒 Invalid API key")
except Exception:
    st.sidebar.warning("🔒 Cannot authenticate")

# Retrain progress
try:
    rs = requests.get(f"{API_URL}/retrain/status", headers=HEADERS, timeout=3)
    if rs.status_code == 200:
        status = rs.json()
        st.sidebar.divider()
        st.sidebar.caption("**Auto-retrain**")
        st.sidebar.progress(
            min(1.0, status["progress_pct"] / 100),
            text=f"{status['upload_count']} / {status['threshold']} uploads",
        )
        if status.get("last_train"):
            st.sidebar.caption(f"Last train: {str(status['last_train'])[:19]}")
except Exception:
    pass

# Navigation
st.sidebar.divider()
page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Forecast",
        "📥 Upload POS Data",
        "📋 Upload Summary",
        "🔬 Data Quality",
        "⚙️ Admin",
    ],
)


# ============================================================
# PAGE: Forecast
# ============================================================
if page == "📊 Forecast":
    st.title("📊 Sales Forecast")
    col1, col2, col3 = st.columns(3)
    with col1:
        store_id = st.number_input("Store ID", min_value=1, value=1)
    with col2:
        sku_id = st.number_input("SKU ID", min_value=1, value=1)
    with col3:
        horizon = st.slider("Horizon (days)", 1, 90, 7)

    if st.button("🔮 Generate Forecast", type="primary"):
        with st.spinner("Calling forecast API..."):
            resp = requests.post(
                f"{API_URL}/forecast",
                params={"store_id": store_id, "sku_id": sku_id, "horizon": horizon},
                headers=HEADERS,
            )
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame(data["predictions"])
            df["date"] = pd.to_datetime(df["date"])
            st.success(
                f"Forecast for store={store_id}, sku={sku_id} "
                f"(baseline ≈ {data.get('baseline_daily_sales', '?')} units/day)"
            )
            st.line_chart(df.set_index("date")["forecast"])
            st.dataframe(df, use_container_width=True)
            st.metric("Total forecast units", f"{df['forecast'].sum():,.0f}")
        else:
            st.error(f"Error {resp.status_code}: {resp.text}")


# ============================================================
# PAGE: Upload POS Data
# ============================================================
elif page == "📥 Upload POS Data":
    st.title("📥 Upload POS Data")

    if not USER_ROLE:
        st.error("You need a valid API key to upload data.")
        st.stop()

    if USER_ROLE not in ["admin", "analyst"]:
        st.warning(f"Your role '{USER_ROLE}' cannot upload data.")
        st.stop()

    tab1, tab2 = st.tabs(["✍️ Manual Entry", "📄 Bulk File Upload"])

    # ----- Manual -----
    with tab1:
        st.subheader("Enter a Single Sale")
        with st.form("manual_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                m_date = st.date_input("Date", value=date.today())
                m_store = st.number_input("Store ID", min_value=1, value=1)
            with c2:
                m_sku = st.number_input("SKU ID", min_value=1, value=1)
                m_qty = st.number_input("Sales Units", min_value=0, value=10)
            with c3:
                m_price = st.number_input("Price", min_value=0.0, value=19.99, step=0.01)
                m_promo = st.checkbox("On Promotion", value=False)

            m_inv = st.number_input("Inventory Level", min_value=0, value=100)
            submitted = st.form_submit_button("Submit Record", type="primary")

        if submitted:
            files = {
                "date": (None, m_date.isoformat()),
                "store_id": (None, str(m_store)),
                "sku_id": (None, str(m_sku)),
                "sales_units": (None, str(m_qty)),
                "price": (None, str(m_price)),
                "promotion_flag": (None, str(int(m_promo))),
                "inventory_level": (None, str(m_inv)),
            }
            r = requests.post(f"{API_URL}/upload/manual", files=files, headers=HEADERS)
            if r.status_code == 200:
                st.success(f"✅ {r.json()['message']}")
                q = r.json().get("quality", {})
                for w in q.get("warnings", []):
                    st.warning(w)
            else:
                st.error(f"Error {r.status_code}: {r.text}")

    # ----- Bulk -----
    with tab2:
        st.subheader("Upload CSV or Excel File")
        st.info(
            "**Required columns:** `date`, `store_id`, `sku_id`, `sales_units`\n\n"
            "**Optional:** `price`, `promotion_flag`, `inventory_level`\n\n"
            "**Supported:** `.csv`, `.xlsx`, `.xls`"
        )

        sample = pd.DataFrame({
            "date": ["2026-09-10", "2026-09-10", "2026-09-11"],
            "store_id": [1, 1, 1],
            "sku_id": [1, 2, 1],
            "sales_units": [45, 30, 50],
            "price": [19.99, 29.99, 19.99],
            "promotion_flag": [0, 1, 0],
            "inventory_level": [120, 80, 115],
        })
        st.download_button(
            "⬇️ Download Sample CSV",
            sample.to_csv(index=False).encode("utf-8"),
            "sample_pos_upload.csv",
            "text/csv",
        )

        uploaded = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    df_preview = pd.read_csv(uploaded)
                else:
                    df_preview = pd.read_excel(uploaded)
                st.write("**Preview:**")
                st.dataframe(df_preview.head(20), use_container_width=True)
                st.caption(f"{len(df_preview)} rows, {len(df_preview.columns)} columns")
            except Exception as e:
                st.error(f"Cannot preview file: {e}")
                df_preview = None

            if df_preview is not None and st.button("🚀 Upload to API", type="primary"):
                uploaded.seek(0)
                files = {"file": (uploaded.name, uploaded.getvalue())}
                with st.spinner("Uploading..."):
                    r = requests.post(
                        f"{API_URL}/upload/csv", files=files, headers=HEADERS
                    )
                if r.status_code == 200:
                    resp = r.json()
                    st.success(
                        f"✅ Uploaded {resp['rows_received']} rows "
                        f"({resp['rows_valid']} valid) from {resp['file_type']}"
                    )

                    q = resp.get("quality", {})
                    if q:
                        st.subheader("🔬 Quality Report")
                        c1, c2 = st.columns(2)
                        c1.metric("Quality Score", f"{q.get('quality_score', 0)}/100")
                        c2.metric("Warnings", len(q.get("warnings", [])))
                        for w in q.get("warnings", []):
                            st.warning(w)
                        with st.expander("Detailed issues"):
                            st.json(q.get("issues", {}))

                    rt = resp.get("retrain", {})
                    if rt.get("triggered"):
                        st.info("🚀 Auto-retrain triggered! Training in background.")
                    elif rt:
                        st.caption(
                            f"Auto-retrain progress: {rt.get('upload_count', 0)} / "
                            f"{rt.get('threshold', 0)}"
                        )
                else:
                    st.error(f"Error {r.status_code}: {r.text}")


# ============================================================
# PAGE: Upload Summary
# ============================================================
elif page == "📋 Upload Summary":
    st.title("📋 Upload Summary")

    r = requests.get(f"{API_URL}/upload/summary", headers=HEADERS)
    if r.status_code == 200:
        data = r.json()
        if data["total_rows"] == 0:
            st.warning("No POS uploads yet.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Rows", f"{data['total_rows']:,}")
            c2.metric("Stores", data["n_stores"])
            c3.metric("SKUs", data["n_skus"])
            c4.metric("Sales Units", f"{data['total_sales_units']:,}")
            st.write(
                f"**Date range:** {data['date_range']['min']} → "
                f"{data['date_range']['max']}"
            )
            st.write(f"**Last updated:** {data['last_updated']}")

            r2 = requests.get(
                f"{API_URL}/upload/preview?limit=50", headers=HEADERS
            )
            if r2.status_code == 200:
                st.subheader("Latest Uploads")
                st.dataframe(
                    pd.DataFrame(r2.json()["rows"]), use_container_width=True
                )
    else:
        st.error(f"Error {r.status_code}: {r.text}")


# ============================================================
# PAGE: Data Quality
# ============================================================
elif page == "🔬 Data Quality":
    st.title("🔬 Data Quality Report")

    if st.button("🔄 Refresh", type="primary"):
        st.rerun()

    r = requests.get(f"{API_URL}/upload/preview?limit=5000", headers=HEADERS)
    if r.status_code != 200:
        st.error("Cannot fetch data for analysis")
    else:
        rows = r.json().get("rows", [])
        if not rows:
            st.info("No uploaded data to analyze yet.")
        else:
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])

            n = len(df)
            n_dups = df.duplicated(subset=["date", "store_id", "sku_id"]).sum()
            n_neg = (df["sales_units"] < 0).sum()
            n_future = (df["date"] > pd.Timestamp.now()).sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{n:,}")
            c2.metric("Duplicates", int(n_dups))
            c3.metric("Negative Sales", int(n_neg))
            c4.metric("Future Dates", int(n_future))

            st.subheader("Summary by Store-SKU")
            summary = (
                df.groupby(["store_id", "sku_id"])
                .agg(
                    rows=("sales_units", "count"),
                    total=("sales_units", "sum"),
                    avg=("sales_units", "mean"),
                    std=("sales_units", "std"),
                )
                .round(2)
                .reset_index()
            )
            st.dataframe(summary, use_container_width=True)

            st.subheader("Sales by Store")
            st.bar_chart(df.groupby("store_id")["sales_units"].sum())


# ============================================================
# PAGE: Admin
# ============================================================
elif page == "⚙️ Admin":
    st.title("⚙️ Admin Panel")

    if USER_ROLE != "admin":
        st.warning("Admin access required.")
        st.stop()

    st.subheader("Auto-Retrain Control")
    r = requests.get(f"{API_URL}/retrain/status", headers=HEADERS)
    if r.status_code == 200:
        s = r.json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Uploads counted", s["upload_count"])
        c2.metric("Threshold", s["threshold"])
        c3.metric("Progress", f"{s['progress_pct']}%")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Force Retrain Now", type="primary"):
                rr = requests.post(
                    f"{API_URL}/retrain/trigger", headers=HEADERS
                )
                if rr.status_code == 200:
                    st.success("Retrain triggered in background.")
                else:
                    st.error(rr.text)
        with col2:
            if st.button("🔄 Reset Counter"):
                rr = requests.post(
                    f"{API_URL}/retrain/reset", headers=HEADERS
                )
                if rr.status_code == 200:
                    st.success("Counter reset.")
                else:
                    st.error(rr.text)

        if s.get("history"):
            st.subheader("Recent Retrains")
            st.dataframe(pd.DataFrame(s["history"]), use_container_width=True)

    st.divider()
    st.subheader("Danger Zone")
    if st.button("🗑️ Clear All Uploads"):
        rr = requests.delete(f"{API_URL}/upload/clear", headers=HEADERS)
        if rr.status_code == 200:
            st.success("All uploads cleared.")
        else:
            st.error(rr.text)
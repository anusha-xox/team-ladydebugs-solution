import os
from datetime import datetime
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from google.cloud import bigquery, aiplatform
from vertexai import init as vertex_init
from vertexai.generative_models import GenerativeModel
USE_GEMINI = True
from dotenv import load_dotenv
import os, base64
load_dotenv()

st.set_page_config(page_title="AI Governance • Health Forecasts", page_icon="🩺", layout="wide")

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("DATASET_ID")
LOCATION = os.getenv("LOCATION")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
# b64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")
# if b64:
#     sa_path = "sa.json"
#     with open(sa_path, "w") as f:
#         f.write(base64.b64decode(b64).decode())

TBL_LONG = f"`{PROJECT_ID}.{DATASET_ID}.hmis_solapur_long`"
TBL_TS = f"`{PROJECT_ID}.{DATASET_ID}.immunisation_ts`"
TBL_FC_NEXT = f"`{PROJECT_ID}.{DATASET_ID}.immunisation_forecast_next`"
VIEW_TOP5 = f"`{PROJECT_ID}.{DATASET_ID}.v_top5_immunisation_spikes`"

REQUIRED = {"PROJECT_ID": PROJECT_ID, "DATASET_ID": DATASET_ID}
missing = [k for k,v in REQUIRED.items() if not v]
if missing:
    st.error(f"Missing env/secrets for: {', '.join(missing)}. "
             "Set them in .env or .streamlit/secrets.toml")
    st.stop()


from google.cloud import bigquery

@st.cache_data(show_spinner=False, ttl=300)
def bq_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    client = bigquery.Client(project=PROJECT_ID)
    job_config = bigquery.QueryJobConfig()

    if params:
        qparams = []
        for name, (ptype, value) in params.items():
            if ptype.upper().startswith("ARRAY<"):  
                elem_type = ptype[6:-1]             
                qparams.append(bigquery.ArrayQueryParameter(name, elem_type, value))
            else:
                qparams.append(bigquery.ScalarQueryParameter(name, ptype, value))
        job_config.query_parameters = qparams

    return client.query(sql, job_config=job_config).result().to_dataframe()


def kpi_card(col, label, value, help_text=None):
    col.metric(label, value if value is not None else "—", help=help_text)

def month_order():
    return ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

st.sidebar.title("⚙️ Controls")
st.sidebar.caption("Data source: BigQuery HMIS (Solapur, 12 months).")

items_df = bq_df(
    f"""
    SELECT DISTINCT item
    FROM {TBL_LONG}
    ORDER BY item
    """
)
default_item = "M9 [CHILD IMMUNISATION]"
item_choice = st.sidebar.selectbox("Indicator", items_df["item"], index=int(np.where(items_df["item"]==default_item)[0][0]) if default_item in items_df["item"].values else 0)

subs_df = bq_df(
    f"""
    SELECT DISTINCT subdistrict
    FROM {TBL_LONG}
    WHERE subdistrict IS NOT NULL
    ORDER BY subdistrict
    """
)
subdistrict_multi = st.sidebar.multiselect("Subdistricts", subs_df["subdistrict"].tolist(), default=subs_df["subdistrict"].tolist()[:6])

st.sidebar.divider()
# st.sidebar.write("Auth:", os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "ADC (gcloud)"))
st.sidebar.write("Project:", PROJECT_ID)

st.title("🩺 AI-Powered Health Forecasts • Solapur (Pilot)")
st.caption("Predictive governance prototype using BigQuery ML + Streamlit")

kpi_sql = f"""
WITH monthly AS (
  SELECT
    subdistrict,
    Month_File,
    SUM(value) AS total_value
  FROM {TBL_LONG}
  WHERE segment='Total' AND item=@item
  GROUP BY subdistrict, Month_File
),
ts AS (
  SELECT
    subdistrict,
    SAFE_CAST(STRPOS('JanFebMarAprMayJunJulAugSepOctNovDec', SUBSTR(Month_File,1,3))/3 + 1 AS INT64) AS month_num,
    SUM(total_value) AS total_value
  FROM monthly
  GROUP BY subdistrict, Month_File
),
last_actual AS (
  SELECT subdistrict, ANY_VALUE(total_value) OVER (PARTITION BY subdistrict) AS last_actual_value
  FROM (
    SELECT subdistrict, total_value,
           ROW_NUMBER() OVER (PARTITION BY subdistrict ORDER BY month_num DESC) rn
    FROM ts
  )
  WHERE rn=1
),
fc_next AS (
  -- use the immunisation_forecast_next by default; if user chose another item, compute quick heuristic
  SELECT subdistrict, forecast_value AS predicted_value
  FROM {TBL_FC_NEXT}
)
SELECT
  SUM(l.last_actual_value) AS total_actual_last,
  SUM(COALESCE(f.predicted_value, l.last_actual_value)) AS total_predicted_next
FROM last_actual l
LEFT JOIN fc_next f USING(subdistrict)
"""
kpis = bq_df(kpi_sql, params={"item": ("STRING", item_choice)})
c1, c2, c3 = st.columns(3)
kpi_card(c1, "Last Month Total", f"{int(kpis['total_actual_last'][0]):,}")
kpi_card(c2, "Predicted Next Month", f"{int(kpis['total_predicted_next'][0]):,}")
growth = (kpis['total_predicted_next'][0] - kpis['total_actual_last'][0]) / max(kpis['total_actual_last'][0], 1)
kpi_card(c3, "Growth vs Last Month", f"{growth*100:,.1f}%")

st.divider()

st.subheader("📌 Top 5 subdistricts: expected change next month (Child Immunisation)")
top5 = bq_df(f"SELECT * FROM {VIEW_TOP5}")
left, mid = st.columns([1,2])
with left:
    st.dataframe(top5.rename(columns={
        "subdistrict":"Subdistrict",
        "last_actual_value":"Last month",
        "predicted_value":"Forecast",
        "abs_increase":"Δ change"
    }), use_container_width=True, hide_index=True)
with mid:
    barch = alt.Chart(top5).mark_bar().encode(
        y=alt.Y('subdistrict:N', sort='-x', title="Subdistrict"),
        x=alt.X('abs_increase:Q', title="Absolute change (next vs last)"),
        tooltip=['subdistrict','last_actual_value','predicted_value','abs_increase']
    ).properties(height=260)
    st.altair_chart(barch, use_container_width=True)

st.divider()

st.subheader(f"📈 Monthly trend • {item_choice}")
if len(subdistrict_multi) == 0:
    st.info("Select at least one subdistrict in the sidebar.")
else:
    ts_sql = f"""
    WITH monthly AS (
      SELECT
        subdistrict,
        Month_File,
        SUM(value) AS total_value
      FROM {TBL_LONG}
      WHERE segment='Total' AND item=@item AND subdistrict IN UNNEST(@subs)
      GROUP BY subdistrict, Month_File
    )
    SELECT
      subdistrict,
      Month_File,
      total_value
    FROM monthly
    """
    tsdf = bq_df(ts_sql, params={
        "item": ("STRING", item_choice),
        "subs": ("ARRAY<STRING>", subdistrict_multi)
    })
    tsdf['Month_File'] = pd.Categorical(tsdf['Month_File'], categories=month_order(), ordered=True)
    tsdf = tsdf.sort_values(['subdistrict','Month_File'])
    line = alt.Chart(tsdf).mark_line(point=True).encode(
        x=alt.X('Month_File:N', title="Month"),
        y=alt.Y('total_value:Q', title="Total"),
        color='subdistrict:N',
        tooltip=['subdistrict','Month_File','total_value']
    ).properties(height=320)
    st.altair_chart(line, use_container_width=True)

st.divider()
st.subheader("🧠 AI Insight (optional)")
user_q = st.text_input("Ask for a summary (e.g., 'Which subdistrict needs most attention next month and why?')", "")
if st.button("Generate Insight") or user_q:
    try:
        ctx = top5[['subdistrict','last_actual_value','predicted_value','abs_increase']].copy()
        ctx = ctx.rename(columns={'subdistrict':'Subdistrict','last_actual_value':'LastMonth','predicted_value':'Forecast','abs_increase':'Change'})
        ctx_json = ctx.to_dict(orient='records')

        prompt = f"""
You are an AI Health Governance advisor. Using the JSON below (top 5 subdistrict changes for next month) and the indicator "{item_choice}",
write 3 concise, actionable recommendations for district officials. Be specific and avoid jargon.

JSON:
{ctx_json}
"""
        if USE_GEMINI:
            vertex_init(project=PROJECT_ID, location=LOCATION)
            model = GenerativeModel(GEMINI_MODEL)
            resp = model.generate_content(prompt)
            st.success(resp.text)
        else:
            df_ = top5.sort_values('abs_increase', ascending=False).reset_index(drop=True)
            lines = []
            for i, r in df_.iterrows():
                direction = "increase" if r['abs_increase'] >= 0 else "drop"
                lines.append(f"- {r['subdistrict']}: {direction} of {abs(int(r['abs_increase'])):,} vs last month; plan vaccine stock and staff accordingly.")
                if i == 2: break
            st.info("\n".join(lines) + "\n\n(Enable USE_GEMINI=true for richer summaries.)")
    except Exception as e:
        st.error(f"Insight generation failed: {e}")

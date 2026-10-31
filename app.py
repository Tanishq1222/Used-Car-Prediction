
import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "cleaned_cars.csv"
MODEL_PATH = ROOT / "model" / "car_price_pipeline.pkl"
BG_PATH = ROOT / "assets" / "car_background.jpg"
LOGO_PATH = ROOT / "assets" / "logo.png"

# Palette taken from the background photo: night blue, headlight, brake caliper
INK = "#e8eef7"
MUTED = "#93a4bd"
HEADLIGHT = "#9bd8ff"
CALIPER = "#e5484d"
PANEL = "rgba(7, 13, 24, 0.76)"

FEATURES = [
    "brand", "vehicle_age", "km_driven", "seller_type", "fuel_type",
    "transmission_type", "mileage", "engine", "max_power", "seats",
]

st.set_page_config(
    page_title="Used Car Price Estimator",
    page_icon="🚗",
    layout="wide",
)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_pipeline():
    """Load the saved pipeline, or return None if it has not been created yet."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    df["price_lakh"] = df["selling_price"] / 1e5  # charts show prices in lakh
    return df


@st.cache_data
def background_css(path_str: str) -> str:
    """Return CSS that sets the car photo as a fixed full-page background."""
    path = Path(path_str)
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"""
    .stApp {{
        background-image:
            linear-gradient(90deg, rgba(5,9,18,0.80) 0%, rgba(5,9,18,0.22) 55%, rgba(5,9,18,0.40) 100%),
            url("data:image/jpeg;base64,{b64}");
        background-size: cover;
        background-position: 72% center;
        background-attachment: fixed;
    }}
    """


@st.cache_data
def logo_html(path_str: str) -> str:
    """Return the sidebar logo badge as HTML, or an empty string if the file is missing."""
    path = Path(path_str)
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<div class="brand"><img src="data:image/png;base64,{b64}" alt="Used Car Price Prediction"></div>'


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def inr(x: float) -> str:
    """Format rupees the way Indian car listings do (lakh / crore)."""
    if x >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if x >= 1e5:
        return f"₹{x / 1e5:.2f} L"
    return f"₹{x:,.0f}"


def predict_with_range(pipeline, rows: pd.DataFrame):
    """Return (estimate, low, high) for each row.

    The estimate is the forest's average. The range is the 10th-90th percentile
    of the individual trees' predictions, so it shows how much the trees disagree.
    """
    pre = pipeline.named_steps["preprocessor"]
    forest = pipeline.named_steps["model"]
    Xt = pre.transform(rows[FEATURES])
    Xt = Xt.tocsr().astype(np.float32) if hasattr(Xt, "tocsr") else Xt.astype(np.float32)
    per_tree = np.vstack([tree.predict(Xt) for tree in forest.estimators_])
    return (
        per_tree.mean(axis=0),
        np.percentile(per_tree, 10, axis=0),
        np.percentile(per_tree, 90, axis=0),
    )


def style_fig(fig, height=330):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        font=dict(color=INK),
        title_font=dict(size=16),
    )
    return fig


def input_warnings(car: dict, df: pd.DataFrame, n_model_listings: int) -> list[str]:
    """Plain-language checks on the inputs, so odd values don't pass silently."""
    notes = []

    fuel_count = int((df["fuel_type"] == car["fuel_type"]).sum())
    if fuel_count < 50:
        notes.append(
            f"The training data has only {fuel_count} {car['fuel_type']} cars, "
            "so this estimate is very rough."
        )
    if n_model_listings < 10:
        notes.append(
            f"Only {n_model_listings} listings of this model exist in the training data; "
            "treat the estimate with extra caution."
        )

    checks = [
        ("vehicle_age", "Vehicle age", "years"),
        ("km_driven", "Kilometres driven", "km"),
        ("mileage", "Mileage", "km/l"),
        ("engine", "Engine size", "cc"),
        ("max_power", "Max power", "bhp"),
    ]
    for col, label, unit in checks:
        lo, hi = df[col].quantile([0.01, 0.99])
        if not lo <= car[col] <= hi:
            notes.append(
                f"{label} ({car[col]:,.0f} {unit}) is outside the range of most cars "
                f"in the data ({lo:,.0f}-{hi:,.0f} {unit})."
            )

    if car["vehicle_age"] > 0 and car["km_driven"] / car["vehicle_age"] > 40000:
        notes.append("That is over 40,000 km per year, which is unusually heavy use.")
    return notes


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&display=swap');

        {background_css(str(BG_PATH))}

        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{
            background: rgba(5, 9, 18, 0.86);
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(155, 216, 255, 0.12);
        }}
        .block-container {{ padding-top: 2.5rem; max-width: 1250px; }}

        /* The logo is drawn for a light background (dark navy lettering), so it sits on a light badge. */
        .brand {{
            background: linear-gradient(160deg, #f7faff 0%, #dde9fa 100%);
            border-radius: 14px;
            padding: 0.55rem 0.8rem 0.45rem 0.8rem;
            margin: 0 0 1.3rem 0;
            box-shadow: 0 0 26px rgba(155, 216, 255, 0.20);
        }}
        .brand img {{ width: 100%; height: auto; display: block; }}

        h1, h2, h3, .hero-title, .price-figure {{
            font-family: 'Barlow Condensed', 'Arial Narrow', 'Roboto Condensed', sans-serif !important;
            letter-spacing: 0.01em;
        }}

        .hero {{ min-height: 24vh; padding: 1.5rem 0 1.25rem 0; max-width: 640px; }}
        .hero .hero-title {{
            font-size: 3.6rem !important; line-height: 1.02 !important; font-weight: 700 !important;
            color: {INK}; margin: 0 0 0.6rem 0;
        }}
        .hero .hero-sub {{ color: {MUTED}; font-size: 1.05rem !important; margin: 0; }}

        /* No backdrop-filter, filter or transform on the panel: any of them makes it the
           containing block for position:fixed children, which breaks the fullscreen view
           of the charts inside it. A plain translucent background is safe. */
        [data-testid="stTabs"] [role="tabpanel"] {{
            background: {PANEL};
            border: 1px solid rgba(155, 216, 255, 0.14);
            border-radius: 12px;
            padding: 1.4rem 1.4rem 1.2rem 1.4rem;
            margin-top: 0.6rem;
        }}
        button[data-baseweb="tab"] {{ font-size: 1rem; }}

        .price-block .price-label {{ color: {MUTED}; font-size: 0.95rem !important; margin: 0; }}
        .price-block .price-figure {{
            font-size: 5rem !important; font-weight: 700 !important; line-height: 1.05 !important;
            color: {HEADLIGHT}; margin: 0.1rem 0 0.2rem 0;
            text-shadow: 0 0 28px rgba(155, 216, 255, 0.35);
        }}
        .price-block .price-range {{ color: {INK}; font-size: 1.1rem !important; margin: 0 0 0.6rem 0; }}
        .price-block .price-note {{ color: {MUTED}; font-size: 0.85rem !important; margin: 0 0 1rem 0; }}

        [data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.04);
            border-left: 3px solid {HEADLIGHT};
            border-radius: 6px;
            padding: 0.6rem 0.9rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
inject_css()

pipeline = load_pipeline()
if not DATA_PATH.exists():
    st.error(f"Data file not found: `{DATA_PATH.relative_to(ROOT)}`.")
    st.stop()
if pipeline is None:
    st.error(
        "Model file not found. Run `Notebook/model_training.ipynb` to create "
        "`model/car_price_pipeline.pkl`, then restart the app."
    )
    st.stop()

df = load_data()

if "saved" not in st.session_state:
    st.session_state.saved = []

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">What is your used car worth?</div>
        <div class="hero-sub">Describe the car in the sidebar. The estimate updates as you change anything.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar: the car description ----------------------------------------
with st.sidebar:
    st.markdown(logo_html(str(LOGO_PATH)), unsafe_allow_html=True)
    st.header("Your car")

    brands = sorted(df["brand"].unique())
    brand = st.selectbox("Brand", brands, index=brands.index("Maruti") if "Maruti" in brands else 0)

    models = df.loc[df["brand"] == brand, "model"].value_counts().index.tolist()
    model_name = st.selectbox(
        "Model",
        models,
        help="Used only to pre-fill typical specs below. The estimate itself uses the brand and the specs.",
    )

    ref = df[(df["brand"] == brand) & (df["model"] == model_name)]
    sfx = f"{brand}|{model_name}"  # changing model resets the pre-filled defaults

    fuel_options = df["fuel_type"].value_counts().index.tolist()
    trans_options = df["transmission_type"].value_counts().index.tolist()
    seller_options = df["seller_type"].value_counts().index.tolist()

    fuel_type = st.selectbox(
        "Fuel type", fuel_options,
        index=fuel_options.index(ref["fuel_type"].mode().iloc[0]), key=f"fuel|{sfx}",
    )
    transmission_type = st.selectbox(
        "Transmission", trans_options,
        index=trans_options.index(ref["transmission_type"].mode().iloc[0]), key=f"trans|{sfx}",
    )
    seller_type = st.selectbox("Seller type", seller_options)

    vehicle_age = st.slider("Age (years)", 0, 25, 5)
    km_driven = st.slider("Kilometres driven", 0, 300_000, 50_000, step=5_000)

    with st.expander("Specs (pre-filled for this model)", expanded=False):
        mileage = st.number_input(
            "Mileage (km/l)", min_value=float(df["mileage"].min()), max_value=float(df["mileage"].max()),
            value=float(round(ref["mileage"].median(), 1)), step=0.1, key=f"mileage|{sfx}",
        )
        engine = st.number_input(
            "Engine (cc)", min_value=int(df["engine"].min()), max_value=int(df["engine"].max()),
            value=int(ref["engine"].median()), step=50, key=f"engine|{sfx}",
        )
        max_power = st.number_input(
            "Max power (bhp)", min_value=float(df["max_power"].min()), max_value=float(df["max_power"].max()),
            value=float(round(ref["max_power"].median(), 1)), step=1.0, key=f"power|{sfx}",
        )
        seats = st.number_input(
            "Seats", min_value=int(df["seats"].min()), max_value=int(df["seats"].max()),
            value=int(ref["seats"].mode().iloc[0]), step=1, key=f"seats|{sfx}",
        )

car = {
    "brand": brand, "vehicle_age": vehicle_age, "km_driven": km_driven,
    "seller_type": seller_type, "fuel_type": fuel_type,
    "transmission_type": transmission_type, "mileage": mileage,
    "engine": engine, "max_power": max_power, "seats": seats,
}
car_df = pd.DataFrame([car])

est, low, high = (v[0] for v in predict_with_range(pipeline, car_df))

tab_predict, tab_market, tab_saved, tab_about = st.tabs(
    ["Estimate", "Market explorer", "My estimates", "About the model"]
)

# ---- Tab 1: Estimate ------------------------------------------------------
with tab_predict:
    left, right = st.columns([1.05, 1])

    peers = df[(df["brand"] == brand) & (df["vehicle_age"].between(vehicle_age - 1, vehicle_age + 1))]
    peer_label = f"{brand} cars aged {max(vehicle_age - 1, 0)}-{vehicle_age + 1} years"
    if len(peers) < 8:
        peers = df[df["brand"] == brand]
        peer_label = f"all {brand} cars"
    peer_median = float(peers["selling_price"].median())
    pct_vs_peers = (est / peer_median - 1) * 100

    with left:
        st.markdown(
            f"""
            <div class="price-block">
            <div class="price-label">Estimated selling price · {brand} {model_name}</div>
            <div class="price-figure">{inr(est)}</div>
            <div class="price-range">Likely range {inr(low)} to {inr(high)}</div>
            <div class="price-note">The range covers the middle 80% of the model's individual
            decision trees. Real prices can still fall outside it.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)
        m1.metric(
            "Similar-car median",
            inr(peer_median),
            delta=f"{pct_vs_peers:+.0f}% for this car",
            delta_color="off",
            help=f"Median selling price of {peer_label} in the data.",
        )
        m2.metric("Listings compared", f"{len(peers):,}")

        if st.button("Save this estimate", width="stretch"):
            entry = {
                "Car": f"{brand} {model_name}",
                "Age (yrs)": vehicle_age,
                "Km driven": km_driven,
                "Fuel": fuel_type,
                "Transmission": transmission_type,
                "Estimate (₹)": round(float(est)),
                "Low (₹)": round(float(low)),
                "High (₹)": round(float(high)),
            }
            if entry in st.session_state.saved:
                st.toast("This exact estimate is already saved.")
            else:
                st.session_state.saved.append(entry)
                st.toast("Saved. See the My estimates tab.")

        for note in input_warnings(car, df, len(ref)):
            st.warning(note)

    with right:
        fig = px.histogram(
            peers, x="price_lakh", nbins=30,
            title=f"Where this car sits among {peer_label}",
            color_discrete_sequence=["#4a6a8a"],
        )
        fig.add_vrect(x0=low / 1e5, x1=high / 1e5, fillcolor=HEADLIGHT, opacity=0.14, line_width=0)
        fig.add_vline(x=est / 1e5, line_color=CALIPER, line_width=3)
        fig.update_xaxes(title="Selling price (₹ lakh)")
        fig.update_yaxes(title="Listings")
        st.plotly_chart(style_fig(fig, 340), width="stretch")
        st.caption("Red line: your estimate. Shaded band: the likely range.")

    st.divider()

    # Value over the next few years
    st.subheader("How the value may change")
    km_per_year = st.slider("Expected driving per year (km)", 2_000, 30_000, 10_000, step=1_000)

    years = np.arange(0, 6)
    future = pd.concat([car_df] * len(years), ignore_index=True)
    future["vehicle_age"] = np.minimum(vehicle_age + years, 29)
    future["km_driven"] = km_driven + years * km_per_year
    f_est, f_low, f_high = predict_with_range(pipeline, future)
    # A forest predicts in steps, so the raw curve can wobble upward. A car should not gain
    # value as it ages, so keep each year at or below the year before.
    f_est, f_low, f_high = (np.minimum.accumulate(v) for v in (f_est, f_low, f_high))
    retained = f_est[-1] / f_est[0] * 100
    f_est, f_low, f_high = f_est / 1e5, f_low / 1e5, f_high / 1e5  # chart in lakh

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=f_high, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=years, y=f_low, mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(155,216,255,0.16)", name="Likely range", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=years, y=f_est, mode="lines+markers", name="Estimate",
        line=dict(color=HEADLIGHT, width=3), marker=dict(size=8),
        hovertemplate="In %{x} yrs: ₹%{y:.2f} L<extra></extra>",
    ))
    fig.update_xaxes(title="Years from now", dtick=1)
    fig.update_yaxes(title="Estimated price (₹ lakh)")
    fig.update_layout(title="Estimated price if you keep the car")
    st.plotly_chart(style_fig(fig, 320), width="stretch")

    st.caption(
        f"In 5 years the model expects roughly {retained:.0f}% of today's estimate. "
        "This re-runs the model with an older car and more kilometres, and never lets the value rise. "
        "It does not account for "
        "inflation, new models, or market swings, so use it as a guide."
    )

    st.divider()

    # Similar real listings
    st.subheader("Similar listings in the data")
    pool = df[df["brand"] == brand].copy()
    dist = (
        ((pool["vehicle_age"] - vehicle_age) / 3) ** 2
        + ((pool["km_driven"] - km_driven) / 40_000) ** 2
        + ((pool["max_power"] - max_power) / 25) ** 2
        + (pool["fuel_type"] != fuel_type) * 1.5
        + (pool["transmission_type"] != transmission_type) * 1.5
    )
    similar = pool.loc[dist.nsmallest(8).index]
    table = pd.DataFrame({
        "Car": similar["car_name"],
        "Age (yrs)": similar["vehicle_age"],
        "Km driven": similar["km_driven"].map("{:,}".format),
        "Fuel": similar["fuel_type"],
        "Transmission": similar["transmission_type"],
        "Power (bhp)": similar["max_power"],
        "Listed price": similar["selling_price"].map(inr),
    })
    st.dataframe(table, hide_index=True, width="stretch")

# ---- Tab 2: Market explorer ----------------------------------------------
with tab_market:
    top_brands = df["brand"].value_counts().index.tolist()
    chosen = st.multiselect("Brands to compare", sorted(df["brand"].unique()), default=top_brands[:5])
    fuels = st.multiselect("Fuel types", fuel_options, default=fuel_options[:2])

    view = df[df["brand"].isin(chosen) & df["fuel_type"].isin(fuels)]
    if view.empty:
        st.info("Pick at least one brand and one fuel type.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Listings", f"{len(view):,}")
        k2.metric("Median price", inr(view["selling_price"].median()))
        k3.metric("Median age", f"{view['vehicle_age'].median():.0f} yrs")
        k4.metric("Median km", f"{view['km_driven'].median():,.0f}")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(
                view, x="brand", y="price_lakh", color="brand", points=False,
                title="Price spread by brand",
            )
            fig.update_yaxes(title="Selling price (₹ lakh)")
            fig.update_xaxes(title=None)
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 360), width="stretch")
        with c2:
            by_age = (
                view[view["vehicle_age"] <= 15]
                .groupby(["brand", "vehicle_age"])["price_lakh"].median().reset_index()
            )
            fig = px.line(
                by_age, x="vehicle_age", y="price_lakh", color="brand", markers=True,
                title="Median price by age",
            )
            fig.update_xaxes(title="Age (years)")
            fig.update_yaxes(title="Median price (₹ lakh)")
            st.plotly_chart(style_fig(fig, 360), width="stretch")

        c3, c4 = st.columns(2)
        with c3:
            by_trans = view.groupby("transmission_type")["price_lakh"].median().reset_index()
            fig = px.bar(
                by_trans, x="transmission_type", y="price_lakh",
                title="Median price by transmission",
                color_discrete_sequence=[HEADLIGHT],
            )
            fig.update_xaxes(title=None)
            fig.update_yaxes(title="Median price (₹ lakh)")
            st.plotly_chart(style_fig(fig, 320), width="stretch")
        with c4:
            sample = view.sample(min(len(view), 1500), random_state=0)
            fig = px.scatter(
                sample, x="max_power", y="price_lakh", color="fuel_type", opacity=0.6,
                title="Power vs price",
            )
            fig.update_xaxes(title="Max power (bhp)")
            fig.update_yaxes(title="Selling price (₹ lakh)")
            st.plotly_chart(style_fig(fig, 320), width="stretch")

# ---- Tab 3: Saved estimates ----------------------------------------------
with tab_saved:
    if not st.session_state.saved:
        st.info("Nothing saved yet. Set up a car, then press **Save this estimate** on the Estimate tab. "
                "Save a few variations to compare them here.")
    else:
        saved = pd.DataFrame(st.session_state.saved)
        saved.index = [f"#{i + 1}" for i in range(len(saved))]

        labels = saved["Car"] + " (" + saved["Age (yrs)"].astype(str) + " yrs, " \
            + saved["Km driven"].map("{:,}".format) + " km)"
        fig = go.Figure(go.Bar(
            x=saved["Estimate (₹)"] / 1e5, y=saved.index + " " + labels, orientation="h",
            marker_color=HEADLIGHT,
            error_x=dict(
                type="data", symmetric=False, color=MUTED,
                array=(saved["High (₹)"] - saved["Estimate (₹)"]) / 1e5,
                arrayminus=(saved["Estimate (₹)"] - saved["Low (₹)"]) / 1e5,
            ),
            hovertemplate="₹%{x:.2f} L<extra></extra>",
        ))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Estimated price (₹ lakh)")
        fig.update_layout(title="Saved estimates (bars show the likely range)")
        st.plotly_chart(style_fig(fig, max(240, 70 * len(saved))), width="stretch")

        st.dataframe(saved, width="stretch")
        a, b = st.columns(2)
        a.download_button(
            "Download as CSV", saved.to_csv().encode(), "saved_estimates.csv", "text/csv", width="stretch",
        )
        if b.button("Clear all", width="stretch"):
            st.session_state.saved = []
            st.rerun()

# ---- Tab 4: About ---------------------------------------------------------
with tab_about:
    st.subheader("How this works")
    st.markdown(
        f"""
        The estimate comes from a **Random Forest regression** trained on
        **{len(df):,} used-car listings** from CarDekho (duplicates removed).
        It uses ten inputs: brand, age, kilometres driven, seller type, fuel type,
        transmission, mileage, engine size, max power and seats.

        **Accuracy.** On cars held out from training, the model explains about 94% of the
        variation in price, with an average miss of roughly ₹1 lakh.

        **Where it is weaker**
        - Expensive and rare cars. The average miss is several lakh for cars above ₹20 lakh,
          because there are few of them to learn from.
        - Electric, LPG and other rare fuel types, which have very few listings.
        - Anything the inputs do not capture: condition, accident history, service records,
          colour, city, and current market demand.

        **The likely range** is the spread between the 10th and 90th percentile of the
        forest's individual trees. It shows how much the model disagrees with itself,
        and is not a guaranteed interval.

        This is an estimate to start a conversation about price, not a formal valuation.
        """
    )
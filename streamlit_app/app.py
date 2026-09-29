import streamlit as st
import geopandas as gpd
import plotly.express as px
from pathlib import Path

# ---------------------------------------------------------
# Shared map color scale
# ---------------------------------------------------------

OBESITY_COLOR_SCALE = [
    [0.00, "#63BE7B"],  # green
    [0.25, "#A9D86E"],  # yellow-green
    [0.50, "#FFD966"],  # yellow
    [0.70, "#F4A261"],  # orange
    [1.00, "#F8696B"]   # red
]

OBESITY_MIN = 20
OBESITY_MAX = 45

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="County Obesity Prediction",
    page_icon="🗺️",
    layout="wide"
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAP_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "map_final"
    / "final_county_map_data.geojson"
)


# ---------------------------------------------------------
# Load final V2 map data
# ---------------------------------------------------------

@st.cache_data
def load_map_data():
    return gpd.read_file(MAP_FILE)


map_gdf = load_map_data()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("County-Level Adult Obesity Prediction")

st.caption(
    "Predicted adult obesity prevalence across U.S. counties "
    "using the final stacking ensemble."
)


# ---------------------------------------------------------
# Basic verification
# ---------------------------------------------------------

st.write(f"Counties included: **{len(map_gdf):,}**")


# ---------------------------------------------------------
# Choropleth
# ---------------------------------------------------------

geojson = map_gdf.__geo_interface__

fig = px.choropleth(
    map_gdf,
    geojson=geojson,
    locations="FIPS",
    featureidkey="properties.FIPS",
    color="Predicted_Obesity_AdjPrev",
    color_continuous_scale=OBESITY_COLOR_SCALE,
    range_color=(OBESITY_MIN, OBESITY_MAX),
    hover_name="County_Name",
    custom_data=["FIPS"],
    hover_data={
        "State_Name": True,
        "CDC_Obesity_AdjPrev": ":.2f",
        "Predicted_Obesity_AdjPrev": ":.2f",
        "Prediction_Difference": ":.2f",
        "FIPS": False
    },
    labels={
        "Predicted_Obesity_AdjPrev": "Predicted Obesity (%)",
        "CDC_Obesity_AdjPrev": "CDC Target (%)",
        "Prediction_Difference": "Difference",
        "State_Name": "State"
    }
)

fig.update_geos(
    fitbounds="locations",
    visible=False
)

fig.update_layout(
    title="Predicted Adult Obesity Prevalence by County",
    margin={
        "r": 0,
        "t": 50,
        "l": 0,
        "b": 0
    },
    height=700
)


# ---------------------------------------------------------
# County detail popup
# ---------------------------------------------------------

@st.dialog("County Details", width="large")
def show_county_details(selected_fips):

    # Get selected county
    selected_county = map_gdf[
        map_gdf["FIPS"].astype(str).str.zfill(5) == selected_fips
    ].copy()

    if selected_county.empty:
        st.error("County data could not be found.")
        return

    county = selected_county.iloc[0]

    # -----------------------------------------------------
    # County heading
    # -----------------------------------------------------

    st.subheader(county["County_Name"])

    st.caption(
        f"FIPS: {county['FIPS']} | "
        f"State: {county['State_Name']}"
    )

    # More space for map
    details_col, map_col = st.columns(
        [0.55, 1.45],
        gap="medium"
    )

    # -----------------------------------------------------
    # LEFT: Prediction summary
    # -----------------------------------------------------

    with details_col:

        st.markdown("### Prediction Summary")

        st.metric(
            "Predicted Prevalence",
            f"{county['Predicted_Obesity_AdjPrev']:.2f}%"
        )

        st.metric(
            "CDC PLACES Target Estimate",
            f"{county['CDC_Obesity_AdjPrev']:.2f}%"
        )

        difference = county["Prediction_Difference"]

        st.metric(
            "Prediction Difference",
            f"{difference:+.2f} pp"
        )

    # -----------------------------------------------------
    # RIGHT: Local map
    # -----------------------------------------------------

    with map_col:

        selected_geom = selected_county.geometry.iloc[0]

        # Get counties directly touching selected county
        neighboring_mask = map_gdf.geometry.touches(
            selected_geom
        )

        nearby_gdf = map_gdf[
            neighboring_mask
            | (
                map_gdf["FIPS"]
                .astype(str)
                .str.zfill(5)
                == selected_fips
            )
        ].copy()

        # Fallback for counties with too few direct neighbors
        if len(nearby_gdf) < 3:

            minx, miny, maxx, maxy = (
                selected_county.total_bounds
            )

            x_padding = max(
                (maxx - minx) * 1.5,
                0.5
            )

            y_padding = max(
                (maxy - miny) * 1.5,
                0.5
            )

            nearby_gdf = map_gdf.cx[
                minx - x_padding : maxx + x_padding,
                miny - y_padding : maxy + y_padding
            ].copy()

        # -------------------------------------------------
        # Base choropleth
        # -------------------------------------------------

        nearby_geojson = nearby_gdf.__geo_interface__

        zoom_fig = px.choropleth(
            nearby_gdf,
            geojson=nearby_geojson,
            locations="FIPS",
            featureidkey="properties.FIPS",
            color="Predicted_Obesity_AdjPrev",
            color_continuous_scale=OBESITY_COLOR_SCALE,
            range_color=(OBESITY_MIN, OBESITY_MAX),

            hover_name="County_Name",

            hover_data={
                "State_Name": True,
                "CDC_Obesity_AdjPrev": ":.2f",
                "Predicted_Obesity_AdjPrev": ":.2f",
                "Prediction_Difference": ":.2f",
                "FIPS": False
            },

            labels={
                "Predicted_Obesity_AdjPrev":
                    "Predicted Obesity (%)",
                "CDC_Obesity_AdjPrev":
                    "CDC Target (%)",
                "Prediction_Difference":
                    "Difference",
                "State_Name":
                    "State"
            }
        )

        # -------------------------------------------------
        # Highlight selected county
        # -------------------------------------------------

        zoom_fig.add_choropleth(
            geojson=selected_county.__geo_interface__,
            locations=selected_county["FIPS"],
            featureidkey="properties.FIPS",

            z=[1],

            colorscale=[
                [0, "rgba(0,0,0,0)"],
                [1, "rgba(0,0,0,0)"]
            ],

            showscale=False,

            marker_line_color="#111111",
            marker_line_width=5,

            hoverinfo="skip"
        )

        # Automatically zoom to local counties
        zoom_fig.update_geos(
            fitbounds="locations",
            visible=False
        )

        # Larger map
        zoom_fig.update_layout(
            height=550,

            margin=dict(
                l=0,
                r=0,
                t=0,
                b=0
            ),

            coloraxis_colorbar=dict(
                title="Predicted<br>Obesity (%)",
                thickness=12,
                len=0.65
            )
        )

        st.plotly_chart(
            zoom_fig,
            width="stretch",
            key=f"detail_map_{selected_fips}"
        )

# ---------------------------------------------------------
# Display main interactive county map
# ---------------------------------------------------------

event = st.plotly_chart(
    fig,
    width="stretch",
    on_select="rerun",
    selection_mode="points",
    key="county_map"
)

# ---------------------------------------------------------
# Handle county selection
# ---------------------------------------------------------

if event and event.selection.points:

    point = event.selection.points[0]

    selected_fips = str(
        point["customdata"][0]
    ).zfill(5)

    st.session_state["selected_fips"] = selected_fips

    show_county_details(selected_fips)

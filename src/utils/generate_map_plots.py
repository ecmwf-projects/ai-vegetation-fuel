"""Inference Plotting Module

This module is used to create (png) map projection plots from ML model inference (csv) files.

Note
-----
The :code:`.csv` files should be named as :code:`{type}_output_{month}.csv`, where :code:`type` is :code:`tropics` or :code:`midlats`.
The csv column headers are expected to be :code:`lat | lon | actual_load | predicted_load`.
"""


import pandas as pd
import geopandas as gpd
import plotly.express as px
from shapely.geometry import Point


def read_csv(csv_file_path: str) -> pd.DataFrame:
    """Reads csv with headers, and creates a Pandas dataframe."""
    df = pd.read_csv(csv_file_path)
    return df


def create_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a new geometry column based on the latitude & longitude positions, and saves the Pandas dataframe."""
    # makes geometry column
    points = df.apply(
        lambda row: Point(row.lat, row.lon), axis=1
    )  # assumes 'lat' and 'lon' as csv column headers for Latitude and Longitude
    graph = gpd.GeoDataFrame(df, geometry=points)
    return df, graph


def generate_plots(
    df_with_geometry: pd.DataFrame,
    output_file_path_act: str,
    output_file_path_pred: str,
    month: str,
) -> None:
    """Creates a .png of actual & predicted map projection plots."""

    def generate_act_plot(df_with_geometry):
        """Plots actual Fuel Load."""
        # assumes 'lat', 'lon' and 'actual_load' as column headers for Latitude, Longitude, and Actual Fuel Load
        fig_act = px.density_mapbox(
            df_with_geometry,
            lat="lat",
            lon="lon",
            z="actual_load",
            radius=3,
            center=dict(lat=0, lon=180),
            zoom=0,
            range_color=[0, 1.592246e11],
            mapbox_style="open-street-map",
            title="Actual Fuel Load Estimate for " + month + " 2016",
        )

        fig_act.write_html(output_file_path_act)

        print(
            "Actual FL plot successfully generated! File saved to ",
            output_file_path_act,
        )

    def generate_pred_plot(df_with_geometry):
        """Plots predicted Fuel Load."""
        # assumes 'lat', 'lon' and 'predicted_load' as column headers for Latitude, Longitude, and Predicted Fuel Load
        fig_pred = px.density_mapbox(
            df_with_geometry,
            lat="lat",
            lon="lon",
            z="predicted_load",
            radius=3,
            center=dict(lat=0, lon=180),
            zoom=0,
            range_color=[0, 1.592246e11],
            mapbox_style="open-street-map",
            title="Predicted Fuel Load Estimate for " + month + " 2016",
        )

        fig_pred.write_html(output_file_path_pred)

        print(
            "Predicted FL plot successfully generated! File saved to ",
            output_file_path_pred,
        )

    if "actual_load" in df_with_geometry.columns:
        generate_act_plot(df_with_geometry)
    generate_pred_plot(df_with_geometry)

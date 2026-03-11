from typing import Tuple
import git
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from wbe.constants import DATA_PATH, GROUP_VARS


def get_storage_metadata() -> Tuple[str]:
    """Get the standard metadata for inclusion in data filenames.

    Returns:
        Time stamp and short commit ID as strings
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_t%H%M%S")
    repo = git.Repo(search_parent_directories=True)
    commit_id = repo.git.rev_parse("--short", "HEAD")
    return ts, commit_id


def get_cdc_wbe_data():
    """Download and store CDC WBE data."""
    url = "https://data.cdc.gov/api/views/j9g8-acpt/rows.csv?accessType=DOWNLOAD"
    data = pd.read_csv(url, index_col="sample_collect_date")
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    outdir = DATA_PATH / "wbe"
    outdir.mkdir(exist_ok=True)
    ts, commit_id = get_storage_metadata()
    filename = f"cdc_data_d{ts}_sha{commit_id}.csv"
    data.to_csv(outdir / filename)


def get_jhu_surveillance_data(
    ind: str,
):
    """Download and store the JHU surveillance data.

    Args:
        ind: Either confirmed or deaths to identify sheet
    """
    url = (
        "https://github.com/CSSEGISandData/COVID-19/raw/refs/heads/master/"
        "csse_covid_19_data/csse_covid_19_time_series/"
        f"time_series_covid19_{ind}_US.csv"
    )
    data = pd.read_csv(url, dtype={"UID": str})
    data.index = data["UID"]
    date_cols = [c for c in data.columns if c.count("/") == 2]
    data = data[date_cols].T
    data.index = pd.to_datetime(data.index, format="%m/%d/%y")
    outdir = DATA_PATH / "jhu"
    outdir.mkdir(exist_ok=True)
    ts, commit_id = get_storage_metadata()
    filename = f"jhu_{ind}_d{ts}_sha{commit_id}.csv"
    data.to_csv(outdir / filename)


def get_jhu_lookup():
    """Download and store JHU UIP to FIPS lookup table."""
    url = (
        "https://github.com/CSSEGISandData/COVID-19/"
        "raw/refs/heads/master/csse_covid_19_data/"
        "UID_ISO_FIPS_LookUp_Table.csv"
    )
    lookup = pd.read_csv(url, index_col="UID", dtype={"FIPS": str})
    lookup = lookup["FIPS"].dropna()
    outdir = DATA_PATH / "jhu"
    outdir.mkdir(exist_ok=True)
    ts, commit_id = get_storage_metadata()
    filename = f"fips_lookup_d{ts}_sha{commit_id}.csv"
    lookup.to_csv(outdir / filename)


def get_jhu_county_data(
    data_filename: str,
) -> pd.DataFrame:
    """Get JHU surveillance data with UIDs mapped to FIPS
    and UIDs with no FIPS mapping dropped.

    Args:
        data_filename: Name of the file containing the data (cases or deaths)

    Returns:
        The mapped, filtered data
    """
    lookup_filename = "fips_lookup_d20260226_t204115_sha8b5688e.csv"
    lookup_path = DATA_PATH / "jhu" / lookup_filename
    lookup = pd.read_csv(lookup_path, index_col=0)["FIPS"].astype(str).str.zfill(5)
    lookup.index = lookup.index.astype(str)

    data = pd.read_csv(DATA_PATH / "jhu" / data_filename, index_col=0)
    data.index = pd.to_datetime(data.index)
    data.columns = data.columns.map(lookup)
    return data.loc[:, ~data.columns.isna()]


def split_concentration_var(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Get new columns for the liquid
    and solid PCR concentrations.

    Args:
        data: The raw data loaded by get_cdc_wbe_data

    Returns:
        The data with the additional columns
    """

    # Get masks
    log_mask = data["pcr_target_units"] == "log10 copies/l wastewater"
    linear_mask = ~log_mask
    solid_mask = data["pcr_target_units"] == "copies/g dry sludge"
    liquid_mask = ~solid_mask

    # Three possible values in the dataset
    log_liquid = log_mask & liquid_mask
    lin_liquid = linear_mask & liquid_mask
    lin_solid = linear_mask & solid_mask

    # Initialise new columns
    data.loc[:, "liquid_pcr_conc"] = np.nan
    data.loc[:, "solid_pcr_conc"] = np.nan

    # Fill
    data.loc[lin_liquid, "liquid_pcr_conc"] = data.loc[
        lin_liquid, "pcr_target_avg_conc"
    ]
    data.loc[log_liquid, "liquid_pcr_conc"] = (
        10.0 ** data.loc[log_liquid, "pcr_target_avg_conc"]
    )
    data.loc[lin_solid, "solid_pcr_conc"] = data.loc[lin_solid, "pcr_target_avg_conc"]

    return data


def group_data(
    sample_type: str,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Get median data over shed and collection date.

    Args:
        sample_type: Whether liquid or solid
        data: The data split into solid or liquid

    Returns:
        The grouped data
    """
    data.sort_values(GROUP_VARS, inplace=True)
    grouped_obs = (
        data.groupby(GROUP_VARS, as_index=False).agg(
            pcr_conc=(f"{sample_type}_pcr_conc", "median"),
            n_raw_rows=(f"{sample_type}_pcr_conc", "size"),
            jurisdict=("wwtp_jurisdiction", "last"),
            fips=("county_fips", "last"),
            pop=("population_served", "last"),
            flow_rate=("pcr_target_flowpop_lin", "median"),
        )
    ).sort_values(GROUP_VARS)
    grouped_obs.index = grouped_obs["sample_collect_date"]
    cols = [c for c in grouped_obs.columns if c != "sample_collect_date"]
    return grouped_obs[cols]

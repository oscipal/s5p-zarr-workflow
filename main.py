import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from pyproj import Transformer, CRS
from shapely import box
import os

from s5p_zarr_workflow import reprojection
from s5p_zarr_workflow.download_cdse import download


def list_files(path):

    datasets = []

    for file in os.listdir(tmp_path):
        path = os.path.join(tmp_path, file)
        try:
            datasets.append(reprojection.reproject_CO(path))
        except (ValueError) as e:
            print(f"Warning file {file}: {e}")
            continue

    return datasets



if __name__ == "__main__":
    tmp_path = download()

    datasets = list_files(tmp_path)
    
    merged = reprojection.merge_mean_by_time(datasets)
    print(merged)
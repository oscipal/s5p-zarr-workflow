import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from pyproj import Transformer, CRS
from shapely import box
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling

def pixel_center_coords(transform, width, height):
    """
    Returns 2D arrays X, Y with the projected coordinates
    of each pixel center given an affine transform.
    """
    # Column indices (0..width-1)
    cols = np.arange(width)
    # Row indices (0..height-1)
    rows = np.arange(height)

    # Convert pixel indices to map coords (center positions)
    xs = transform.c + cols * transform.a + transform.b * 0
    ys = transform.f + rows * transform.e + transform.d * 0

    # Meshgrid to 2D arrays
    #X, Y = np.meshgrid(xs, ys)
    return xs, ys

def reproject_numpy_to_equi7_eu_1km(
    data, lons, lats, nodata=np.nan, resampling="bilinear"
):

    assert data.ndim == 2 and data.shape == (len(lats), len(lons))

    if lats[0] < lats[-1]:
        data = data[::-1, :]
        lats = lats[::-1]

    xres = float(abs(lons[1] - lons[0]))
    yres = float(abs(lats[1] - lats[0]))
    west = float(lons.min())
    north = float(lats.max())
    src_transform = from_origin(west - xres/2, north + yres/2, xres, yres)
    src_crs = "EPSG:4326"

    dst_crs = "EPSG:27704"
    out_res = 10000.0

    left, bottom, right, top = rasterio.transform.array_bounds(
        data.shape[0], data.shape[1], src_transform
    )
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, data.shape[1], data.shape[0],
        left, bottom, right, top, resolution=out_res
    )

    from math import floor
    x0 = floor(dst_transform.c / out_res) * out_res
    y0 = floor(dst_transform.f / out_res) * out_res
    dst_transform = rasterio.Affine(out_res, 0, x0, 0, -out_res, y0)

    dst = np.full((dst_h, dst_w), nodata, dtype=data.dtype)
    reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=getattr(Resampling, resampling),
        src_nodata=nodata,
        dst_nodata=nodata,
    )
    return dst, dst_transform, dst_crs

def merge_mean_by_time(datasets):

    ds_aligned = xr.align(*datasets, join="outer")

    ds_cat = xr.concat(ds_aligned, dim="time")

    ds_out = ds_cat.groupby("time").mean(skipna=True)

    ds_out = ds_out.sortby("time")
    return ds_out


def reproject(path, qa_threshold = 0.5, is_no2 = False):

    ds = xr.open_dataset(path, engine="netcdf4", group = "PRODUCT")

    if is_no2:
        layers = ds["averaging_kernel"].coords["layer"].values

        avg_kernel = xr.Dataset({
            f"var_layer_{int(l)}": ds["averaging_kernel"].sel(layer=l).drop_vars("layer")
            for l in layers
        })

        additional_data = ds[["vertices", "layer", "tm5_constant_a", "tm5_constant_b"]]
        ds = ds.drop_vars(["vertices", "layer", "tm5_constant_a", "tm5_constant_b", "averaging_kernel"])

    transformer = Transformer.from_crs("EPSG:27704", "EPSG:4326", always_xy=True)
    x,y = 5700000, 2100000
    lon_max, lat_max = transformer.transform(x, y)
    x,y = 4200000, 900000
    lon_min, lat_min = transformer.transform(x,y)

    austria_bbox = box(lon_min, lat_min, lon_max, lat_max)
    ds = ds.load()
    ds = ds.set_coords(["longitude", "latitude"])
    delta_time  = ds[["delta_time", "latitude", "longitude"]]

    ds = ds.drop_vars(["delta_time", "time_utc" ])

    mask = (
    (ds["latitude"] >= lat_min) & (ds["latitude"] <= lat_max) &
    (ds["longitude"] >= lon_min) & (ds["longitude"] <= lon_max) & 
    (ds["qa_value"] >= qa_threshold)
    )

    if is_no2:
        avg_kernel_subset = avg_kernel.where(mask, drop=True)
        avg_kernel_subset = avg_kernel_subset.rio.write_crs("EPSG:4326", inplace=False)

    ds_subset = ds.where(mask, drop=True) 

    ds_subset = ds_subset.rio.write_crs("EPSG:4326", inplace=False)

    ds_band = ds_subset.squeeze()

    lat_original = ds_band["latitude"]
    lon_original = ds_band["longitude"]

    lat_min, lat_max = lat_original.min().item(), lat_original.max().item()
    lon_min, lon_max = lon_original.min().item(), lon_original.max().item()
    lat_min, lat_max, lon_min, lon_max

    target_resolution = 0.1

    n_points_lat_full = int(np.ceil((lat_max - lat_min) / target_resolution)) + 1
    n_points_lon_full = int(np.ceil((lon_max - lon_min) / target_resolution)) + 1
    target_lat_full = np.linspace(lat_min, lat_max, n_points_lat_full)
    target_lon_full = np.linspace(lon_min, lon_max, n_points_lon_full)
    lon_grid_full, lat_grid_full = np.meshgrid(target_lon_full, target_lat_full)

    gridded_data = {}

    if is_no2:
        gridded_avg_kernel = {}

        for var in avg_kernel_subset.data_vars:
            gridded = griddata((lon_original.values.flatten(), lat_original.values.flatten()),
                                            avg_kernel_subset[var].values.flatten(),
                                            (lon_grid_full, lat_grid_full),
                                            method="linear")

            gridded_avg_kernel[var] = gridded

    for var in ds_band.data_vars:
        if var == "qa_value":
            gridded = griddata((lon_original.values.flatten(), lat_original.values.flatten()),
                                        ds_band[var].values.flatten(),
                                        (lon_grid_full, lat_grid_full),
                                        method="nearest")
            
        else:
            gridded = griddata((lon_original.values.flatten(), lat_original.values.flatten()),
                                            ds_band[var].values.flatten(),
                                            (lon_grid_full, lat_grid_full),
                                            method="linear")

        gridded_data[var] = gridded

    reprojected = {}

    if is_no2:
        reprojected_avg_kernel = {}

        for var in gridded_avg_kernel:
            arr_avg_10km, tr_avg_1km, crs_avg_1km = reproject_numpy_to_equi7_eu_1km(
                gridded_avg_kernel[var], target_lon_full, target_lat_full,
                nodata=np.nan,
                resampling="bilinear"          
            )
            reprojected_avg_kernel[var] = (("y", "x"),arr_avg_10km)

    for var in gridded_data:
        if var == "qa_value":
            arr_10km, tr_1km, crs_1km = reproject_numpy_to_equi7_eu_1km(
                gridded_data[var], target_lon_full, target_lat_full,
                nodata=np.nan,
                resampling="nearest"          
            )
        else:
            arr_10km, tr_1km, crs_1km = reproject_numpy_to_equi7_eu_1km(
                gridded_data[var], target_lon_full, target_lat_full,
                nodata=np.nan,
                resampling="bilinear"          
            )

        reprojected[var] = (("y", "x"),arr_10km)

    xs, ys = pixel_center_coords(tr_1km, arr_10km.shape[1], arr_10km.shape[0])

    sensing_time = delta_time["delta_time"].values[0][0].astype("datetime64[D]")
    
    ds_equi7 = xr.Dataset(reprojected,
                      coords={
                                "x":("x", xs),
                              "y":("y", ys),
                              })
    
    if is_no2:
        ds_avg_kernel_equi7 =  xr.Dataset(reprojected_avg_kernel,
                        coords={
                                "x":("x", xs),
                                "y":("y", ys),
                                })
        slices = [ds_avg_kernel_equi7[f"var_layer_{int(i)}"] for i in layers]

        var3d = xr.concat(slices, dim="layers").assign_coords(layers=layers)

        var3d = var3d.transpose("y", "x", "layers")

        ds_merged = ds_avg_kernel_equi7.assign(averaging_kernel=var3d)
        ds_equi7 = xr.merge([ds_equi7, ds_merged["averaging_kernel"], additional_data], join="exact")
    
    ds_aut = ds_equi7.sel(x=slice(4500000,5390000), y=slice(1800000,1210000)).expand_dims({"time": [sensing_time]})

    return ds_aut
"""Simple SPAD raw reader utilities."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

try:
    from .SPADconfig import (
        DEFAULT_ACTIVE_POINT,
        DEFAULT_DATA_DIRS,
        DEFAULT_PAGES_PER_GROUP,
        DEFAULT_TIME_THRESHOLD,
    )
except ImportError:
    from SPADconfig import (
        DEFAULT_ACTIVE_POINT,
        DEFAULT_DATA_DIRS,
        DEFAULT_PAGES_PER_GROUP,
        DEFAULT_TIME_THRESHOLD,
    )


def raw2frame(
    filename: str,
    pages_per_group: int,
    total_pages: int,
    time_threshold: int,
) -> np.ndarray:
    """Read a SPAD raw file and return grouped frames shaped as (G, 4096, P)."""
    num_pixels = 64 * 64
    data_size = 2

    if not os.path.exists(filename):
        raise IOError(f"File not found: {filename}")
    if pages_per_group <= 0:
        raise ValueError("pages_per_group must be a positive integer")
    if total_pages <= 0:
        raise ValueError("total_pages must be a positive integer")
    if time_threshold <= 0:
        raise ValueError("time_threshold must be a positive integer")
    if total_pages % pages_per_group != 0:
        raise ValueError("total_pages must be divisible by pages_per_group")

    file_size = os.path.getsize(filename)
    total_values = file_size // data_size
    max_pages = total_values // num_pixels
    if max_pages == 0:
        return np.empty((0, 4096, pages_per_group), dtype=np.uint16)
    if total_pages > max_pages:
        raise ValueError(f"total_pages {total_pages} exceeds file pages {max_pages}")

    count = int(total_pages) * num_pixels
    data = np.fromfile(filename, dtype=np.uint16, count=count, offset=0)
    if data.size != count:
        raise IOError(f"File too short: expected {count} values, got {data.size}")

    frames = data.reshape((total_pages, 64, 64))
    flat_frames = frames.reshape((total_pages, num_pixels))
    flat_frames[flat_frames > time_threshold] = 0

    num_groups = total_pages // pages_per_group
    grouped = flat_frames.reshape((num_groups, pages_per_group, num_pixels))
    grouped = np.transpose(grouped, (0, 2, 1)).astype(np.uint16, copy=False)
    return grouped


def resolve_usable_total_pages(
    raw_path: str,
    pages_per_group: int,
    total_pages: int | None = None,
) -> int:
    """Resolve a valid total page count that can be grouped cleanly."""
    file_size = os.path.getsize(raw_path)
    total_values = file_size // 2
    max_pages = total_values // (64 * 64)

    if max_pages == 0:
        return 0

    if total_pages is None:
        total_pages = max_pages - (max_pages % pages_per_group)

    if total_pages <= 0:
        raise ValueError("No complete groups are available in this raw file")
    if total_pages > max_pages:
        raise ValueError(f"total_pages {total_pages} exceeds file pages {max_pages}")
    if total_pages % pages_per_group != 0:
        raise ValueError("total_pages must be divisible by pages_per_group")
    return total_pages


def n3_filter(points: np.ndarray, min_count: int) -> np.ndarray:
    """Filter point-cloud rows by duplicate count."""
    unique_points, counts = np.unique(points, axis=0, return_counts=True)
    mask = counts >= min_count
    filtered_points = unique_points[mask]
    filtered_counts = counts[mask]
    return np.column_stack((filtered_points, filtered_counts))


def fit_gaussian_from_hist(
    counts: np.ndarray,
    centers: np.ndarray,
) -> tuple[float, float] | tuple[None, None]:
    """Estimate Gaussian mean and amplitude from a histogram."""
    mask = counts > 0
    if np.count_nonzero(mask) < 3:
        return None, None

    x = centers[mask]
    y = counts[mask].astype(np.float32)
    log_y = np.log(y)

    try:
        a, b, c = np.polyfit(x, log_y, 2)
    except Exception:
        return None, None

    if not np.isfinite(a) or a >= 0:
        mean = float(np.sum(x * y) / np.sum(y))
        amp = float(y.max())
        return mean, amp

    mean = -b / (2 * a)
    amp = float(np.exp(c - (b * b) / (4 * a)))

    if not np.isfinite(mean) or not np.isfinite(amp) or amp <= 0:
        mean = float(np.sum(x * y) / np.sum(y))
        amp = float(y.max())
    return float(mean), float(amp)


def gaussian_fit_maps(frames: np.ndarray, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """Build depth and intensity maps using per-pixel Gaussian fitting."""
    depth_map = np.zeros((64, 64), dtype=np.float32)
    intensity_map = np.zeros((64, 64), dtype=np.float32)
    centers = np.arange(1, threshold, dtype=np.float32)

    for i in range(64):
        for j in range(64):
            values = frames[:, i, j]
            values = values[(values > 0) & (values < threshold)]
            if values.size < 3:
                continue

            counts = np.bincount(values.astype(np.int64), minlength=threshold)
            hist = counts[1:threshold].astype(np.float32)
            if hist.sum() == 0:
                continue

            mean, amp = fit_gaussian_from_hist(hist, centers)
            if mean is None or amp is None:
                continue

            depth_map[i, j] = mean
            intensity_map[i, j] = amp

    return depth_map, intensity_map


def max_count_maps(filtered_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build depth and intensity maps from counted points."""
    intensity_map = np.zeros((64, 64), dtype=np.int32)
    depth_map = np.zeros((64, 64), dtype=np.int32)
    for x, y, z, count in filtered_points:
        xi = int(x) - 1
        yi = int(y) - 1
        if count > intensity_map[xi, yi]:
            intensity_map[xi, yi] = int(count)
            depth_map[xi, yi] = int(z)
    return depth_map, intensity_map


def export_group_to_txt(
    raw_path: str,
    output_txt_path: str,
    group_number: int,
    pages_per_group: int,
    time_threshold: int,
    total_pages: int | None = None,
) -> str:
    """Export one selected group from a raw file into a txt file."""
    if group_number <= 0:
        raise ValueError("group_number must be a positive integer")

    usable_total_pages = resolve_usable_total_pages(
        raw_path=raw_path,
        pages_per_group=pages_per_group,
        total_pages=total_pages,
    )
    grouped = raw2frame(
        raw_path,
        pages_per_group=pages_per_group,
        total_pages=usable_total_pages,
        time_threshold=time_threshold,
    )

    num_groups = grouped.shape[0]
    if group_number > num_groups:
        raise IndexError(f"group_number {group_number} exceeds available groups {num_groups}")

    selected_group = grouped[group_number - 1]
    output_dir = os.path.dirname(output_txt_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    header = "\n".join(
        (
            f"raw_path={raw_path}",
            f"group_number={group_number}",
            f"pages_per_group={pages_per_group}",
            f"total_pages={usable_total_pages}",
            f"time_threshold={time_threshold}",
            f"group_shape={selected_group.shape}",
        )
    )
    np.savetxt(output_txt_path, selected_group, fmt="%d", header=header, comments="# ")
    return output_txt_path


def export_group_to_txt_and_plot_xyzi(
    raw_path: str,
    output_txt_path: str | None,
    group_number: int,
    pages_per_group: int,
    time_threshold: int,
    total_pages: int | None = None,
    active_point: int = 1,
    min_intensity: int = 2,
    point_size: int = 3,
    block: bool = True,
) -> str | None:
    """Export one group to txt (optional) and show an interactive XYZI scatter plot."""
    if group_number <= 0:
        raise ValueError("group_number must be a positive integer")

    usable_total_pages = resolve_usable_total_pages(
        raw_path=raw_path,
        pages_per_group=pages_per_group,
        total_pages=total_pages,
    )
    grouped = raw2frame(
        raw_path,
        pages_per_group=pages_per_group,
        total_pages=usable_total_pages,
        time_threshold=time_threshold,
    )

    num_groups = grouped.shape[0]
    if group_number > num_groups:
        raise IndexError(f"group_number {group_number} exceeds available groups {num_groups}")

    frame = grouped[group_number - 1]

    saved_txt_path = None
    if output_txt_path:
        output_dir = os.path.dirname(output_txt_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        header = "\n".join(
            (
                f"raw_path={raw_path}",
                f"group_number={group_number}",
                f"pages_per_group={pages_per_group}",
                f"total_pages={usable_total_pages}",
                f"time_threshold={time_threshold}",
                f"group_shape={frame.shape}",
            )
        )
        np.savetxt(output_txt_path, frame, fmt="%d", header=header, comments="# ")
        saved_txt_path = output_txt_path

    frames = frame.T.reshape((frame.shape[1], 64, 64))
    mask = (frames > 0) & (frames < time_threshold)
    indices = np.argwhere(mask)
    values = frames[mask]
    xyz = np.column_stack(
        (
            indices[:, 1] + 1,
            indices[:, 2] + 1,
            values.astype(np.uint16, copy=False),
        )
    )
    xyzi = n3_filter(xyz, active_point)

    if xyzi.size > 0:
        intensity_mask = xyzi[:, 3] >= min_intensity
        xyzi = xyzi[intensity_mask]
    # z filter
    z_mask = (xyzi[:, 2] > 50) & (xyzi[:, 2] < 70)
    xyzi = xyzi[z_mask]
    if block:
        plt.ioff()
    else:
        plt.ion()
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"XYZI Scatter (group={group_number})")
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    ax.set_zlabel("X")

    if xyzi.size == 0:
        print("No valid XYZI points to plot")
        plt.show(block=block)
        if not block:
            plt.pause(0.001)
        return saved_txt_path

    scatter = ax.scatter(
        xyzi[:, 1],
        xyzi[:, 2],
        np.abs(xyzi[:, 0]-65),
        c=xyzi[:, 3],
        s=point_size,
        cmap="viridis",
        marker=".",
    )
    fig.colorbar(scatter, ax=ax, label="Intensity")
    plt.show(block=block)
    if not block:
        plt.pause(0.001)
    return saved_txt_path


def process_raw_file(
    raw_path: str,
    pages_per_group: int,
    time_threshold: int,
    active_point: int,
) -> None:
    """Visualize statistics from the last complete group of one raw file."""
    total_pages = resolve_usable_total_pages(raw_path, pages_per_group)
    if total_pages == 0:
        print(f"Skip {raw_path}: not enough pages for one group")
        return

    grouped = raw2frame(
        raw_path,
        pages_per_group=pages_per_group,
        total_pages=total_pages,
        time_threshold=time_threshold,
    )
    last_group = grouped[-1]

    frames = last_group.T.reshape((pages_per_group, 64, 64))
    mask = (frames > 0) & (frames < time_threshold)
    if not np.any(mask):
        print(f"Skip {raw_path}: no valid points after thresholding")
        return

    indices = np.argwhere(mask)
    values = frames[mask]
    point_cloud = np.column_stack(
        (
            indices[:, 1] + 1,
            indices[:, 2] + 1,
            values.astype(np.uint16, copy=False),
        )
    )

    filtered = n3_filter(point_cloud, active_point)
    if filtered.size == 0:
        print(f"Skip {raw_path}: no points after count filter")
        return

    tof_values = values.astype(np.int64, copy=False)
    tof_hist = np.bincount(tof_values, minlength=time_threshold)

    plt.figure(figsize=(8, 4))
    x_axis = np.arange(1, time_threshold)
    plt.bar(x_axis, tof_hist[1:time_threshold], width=1.0, edgecolor="black", align="center")
    plt.title(f"Photon Count Histogram (ToF): {os.path.basename(raw_path)}")
    plt.xlabel("ToF (time bin)")
    plt.ylabel("Counts")

    depth_map, intensity_map = max_count_maps(filtered)

    plt.figure(figsize=(6, 6))
    plt.imshow(intensity_map, cmap="viridis")
    plt.title(f"Intensity (Gaussian Amp): {os.path.basename(raw_path)}")
    plt.colorbar(label="Amplitude")
    plt.xticks([])
    plt.yticks([])

    plt.figure(figsize=(6, 6))
    plt.imshow(depth_map, cmap="viridis", vmin=0, vmax=time_threshold)
    plt.title(f"Depth (Gaussian Mean): {os.path.basename(raw_path)}")
    plt.colorbar(label="Depth")
    plt.xticks([])
    plt.yticks([])


if __name__ == "__main__":
    fa = r"E:\essay\硕士\研一\SPAD数据\0825"
    son = r"2025-08-25_16-50-11_Delay-0_Width-200.raw"
    export_raw_path = os.path.join(fa, son)
    export_group_number = 45
    export_pages_per_group = 500
    export_time_threshold = 150
    export_total_pages = None
    # export_txt_path = os.path.join("temp", "data", f"group_{export_group_number:04d}.txt")
    export_txt_path = os.path.join(fa, "frame", f"{son}-{export_group_number:04d}.txt")
    print("=== Export Group To TXT ===")
    try:
        export_group_to_txt_and_plot_xyzi(
            raw_path=export_raw_path,
            output_txt_path=None,
            group_number=export_group_number,
            pages_per_group=export_pages_per_group,
            time_threshold=export_time_threshold,
            total_pages=export_total_pages,
            active_point=DEFAULT_ACTIVE_POINT,
        )
    except Exception as error:
        print(f"Export group to txt failed: {error}")

    # raw_dir = DEFAULT_DATA_DIRS[-1]
    # default_pages_per_group = DEFAULT_PAGES_PER_GROUP
    # default_time_threshold = DEFAULT_TIME_THRESHOLD
    # default_active_point = DEFAULT_ACTIVE_POINT
    # single_test_file = os.path.join(raw_dir, "2025-08-26_17-32-15_Delay-0_Width-2000.raw")
    # 
    # print("=== Single File Test ===")
    # try:
    #     process_raw_file(
    #         single_test_file,
    #         pages_per_group=default_pages_per_group,
    #         time_threshold=default_time_threshold,
    #         active_point=default_active_point,
    #     )
    #     plt.show()
    # except Exception as error:
    #     print(f"Single file test failed: {error}")

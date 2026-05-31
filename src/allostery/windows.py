from __future__ import annotations


def generate_window_slices(
    num_frames: int,
    window_size: int,
    horizon_size: int,
    stride: int,
) -> list[tuple[slice, slice]]:
    if stride <= 0:
        raise ValueError("stride must be greater than zero")
    if window_size <= 0:
        raise ValueError("window_size must be greater than zero")
    if horizon_size <= 0:
        raise ValueError("horizon_size must be greater than zero")

    windows: list[tuple[slice, slice]] = []
    start = 0
    stop = num_frames - window_size - horizon_size
    while start <= stop:
        past = slice(start, start + window_size)
        future = slice(start + window_size, start + window_size + horizon_size)
        windows.append((past, future))
        start += stride
    return windows

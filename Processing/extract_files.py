#!/usr/bin/env python3
import argparse
import sys
import os
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _read_excel_with_fallbacks(path: str, sheet: Union[str, int, None]) -> Dict[str, pd.DataFrame]:
    """
    Read an Excel file (.xlsx or .xls) and return a dict of {sheet_name: DataFrame}.
    Tries multiple engines to maximize compatibility and produces clear errors if engines are missing.
    """
    sheet_name = sheet if sheet is not None else None  # None means all sheets
    engines_to_try: List[Optional[str]] = [None, "openpyxl", "xlrd", "odf"]
    last_err = None
    for eng in engines_to_try:
        try:
            dfs = pd.read_excel(path, sheet_name=sheet_name, engine=eng)
            if isinstance(dfs, dict):
                return dfs
            elif isinstance(dfs, pd.DataFrame):
                if isinstance(sheet_name, str):
                    return {sheet_name: dfs}
                elif isinstance(sheet_name, int):
                    return {f"sheet_{sheet_name}": dfs}
                else:
                    return {"Sheet1": dfs}
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to read '{path}'. Tried engines {engines_to_try}. Last error: {type(last_err).__name__}: {last_err}"
    )


def _fix_excel_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix Excel files where the actual column names are in the data rows.
    Looks for a row that has non-null values and sets it as the header.
    Specifically handles the case where first column is like 'Recording title:' 
    and actual headers are in row 3.
    """
    if df.empty:
        return df
    
    # Check if first column looks like metadata (e.g., "Recording title:")
    first_col_name = df.columns[0]
    if isinstance(first_col_name, str) and ":" in first_col_name:
        # Look for the row with actual column names
        # Usually it's the row where first column is something like "Time"
        for idx in range(min(10, len(df))):  # Check first 10 rows
            row_vals = df.iloc[idx].tolist()
            first_val = str(row_vals[0]).strip().lower()
            
            # If we find a row starting with "time" or similar, use it as header
            if first_val in ['time', 't', 'sample']:
                # Set this row as column names
                new_columns = df.iloc[idx].tolist()
                # Clean up the column names
                new_columns = [str(col).strip() if pd.notna(col) else f"Col_{i}" 
                              for i, col in enumerate(new_columns)]
                
                # Remove metadata rows and the header row, keep only data
                df = df.iloc[idx+2:].reset_index(drop=True)  # +2 to skip header and units row
                df.columns = new_columns
                
                return df
    
    return df


def _select_columns(df: pd.DataFrame, columns: Optional[List[str]], numeric_only: bool) -> pd.DataFrame:
    """Subset and/or coerce columns by names or indices; optionally keep only numeric columns."""
    if columns:
        resolved_cols: List[str] = []
        for col in columns:
            try:
                idx = int(col)
                if idx < 0 or idx >= df.shape[1]:
                    raise IndexError(f"Column index {idx} out of range [0, {df.shape[1]-1}]")
                resolved_cols.append(df.columns[idx])
            except (ValueError, IndexError):
                if str(col) not in df.columns:
                    raise KeyError(f"Column '{col}' not found in DataFrame columns {list(df.columns)}")
                resolved_cols.append(str(col))
        df = df[resolved_cols]
    if numeric_only:
        df = df.select_dtypes(include=["number"])
    return df


def _handle_missing(df: pd.DataFrame, dropna: bool, fillna: Optional[float]) -> pd.DataFrame:
    """Missing value strategy: drop rows with any NaN or fill NaNs with a numeric value."""
    if dropna and fillna is not None:
        raise ValueError("Choose either --dropna or --fillna, not both.")
    if dropna:
        return df.dropna(axis=0, how="any")
    if fillna is not None:
        return df.fillna(fillna)
    return df


def _scale_array(arr: np.ndarray, mode: str) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
    """
    Scale array columns:
    - 'none': no scaling
    - 'standard': (x - mean) / std
    - 'minmax': (x - min) / (max - min)
    Returns (scaled_array, per-column params).
    """
    params: Dict[str, Tuple[float, float]] = {}
    if mode == "none" or arr.size == 0:
        return arr, params
    X = arr.astype(float, copy=True)
    if mode == "standard":
        means = np.nanmean(X, axis=0)
        stds = np.nanstd(X, axis=0)
        stds_safe = np.where(stds == 0, 1.0, stds)
        X = (X - means) / stds_safe
        for j in range(X.shape[1]):
            params[str(j)] = (float(means[j]), float(stds[j]))
        return X, params
    elif mode == "minmax":
        mins = np.nanmin(X, axis=0)
        maxs = np.nanmax(X, axis=0)
        ranges = maxs - mins
        ranges_safe = np.where(ranges == 0, 1.0, ranges)
        X = (X - mins) / ranges_safe
        for j in range(X.shape[1]):
            params[str(j)] = (float(mins[j]), float(maxs[j]))
        return X, params
    else:
        raise ValueError(f"Unknown scaling mode: {mode}")


def excel_to_numpy(
    paths: List[str],
    sheet: Optional[Union[str, int]],
    header: Optional[int],
    skiprows: int,
    columns: Optional[List[str]],
    numeric_only: bool,
    dropna: bool,
    fillna: Optional[float],
    scale: str,
    output_dir: str,
    save_meta: bool = True
) -> str:
    """Load Excel files into numpy arrays and save as a single .npz archive. Returns the archive path."""
    os.makedirs(output_dir, exist_ok=True)
    all_arrays: Dict[str, np.ndarray] = {}
    metadata: Dict[str, dict] = {}

    # Work on a copy of paths, since we may append from directories
    pending = list(paths)
    i = 0
    while i < len(pending):
        path = pending[i]
        i += 1

        if not os.path.exists(path):
            print(f"[WARN] Skipping missing path: {path}", file=sys.stderr)
            continue

        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for fname in files:
                    if fname.lower().endswith((".xlsx", ".xls")):
                        pending.append(os.path.join(root, fname))
            continue

        # Read file
        try:
            dfs = _read_excel_with_fallbacks(path, sheet)
        except Exception as e:
            print(f"[ERROR] Could not read '{path}': {e}", file=sys.stderr)
            continue

        # Process each sheet
        for sname, df in dfs.items():
            # First, fix the headers if needed (e.g., actual column names are in data rows)
            df = _fix_excel_headers(df)
            
            # If header/skiprows specified, try re-reading with those options
            if header is not None or skiprows > 0:
                try:
                    df = pd.read_excel(path, sheet_name=sname, header=header, skiprows=skiprows)
                except Exception:
                    # Fallback manipulation
                    if header is not None:
                        new_cols = df.iloc[header].tolist()
                        df = df.iloc[header+1:].reset_index(drop=True)
                        df.columns = new_cols
                    if skiprows > 0:
                        df = df.iloc[skiprows:].reset_index(drop=True)

            # Select columns/types
            try:
                df = _select_columns(df, columns, numeric_only)
            except Exception as e:
                print(f"[WARN] In '{os.path.basename(path)}'[{sname}] column selection issue: {e}", file=sys.stderr)
                continue

            # Missing values
            try:
                df = _handle_missing(df, dropna, fillna)
            except Exception as e:
                print(f"[WARN] In '{os.path.basename(path)}'[{sname}] missing-value handling issue: {e}", file=sys.stderr)
                continue

            # To numpy
            try:
                arr = df.to_numpy()
            except Exception as e:
                print(f"[WARN] In '{os.path.basename(path)}'[{sname}] to_numpy() failed: {e}", file=sys.stderr)
                continue

            # Scale
            try:
                arr_scaled, scale_params = _scale_array(arr, scale)
            except Exception as e:
                print(f"[WARN] In '{os.path.basename(path)}'[{sname}] scaling failed: {e}", file=sys.stderr)
                continue

            key = f"{os.path.splitext(os.path.basename(path))[0]}::{sname}"
            all_arrays[key] = arr_scaled
            metadata[key] = {
                "source_file": os.path.abspath(path),
                "sheet": sname,
                "shape": list(arr_scaled.shape),
                "columns": list(map(str, df.columns)),
                "scale_mode": scale,
                "scale_params": scale_params,
                "dropna": dropna,
                "fillna": fillna,
                "numeric_only": numeric_only,
            }

    if not all_arrays:
        raise RuntimeError("No arrays were produced. Check inputs and options.")

    archive_path = os.path.join(output_dir, "excel_arrays.npz")
    np.savez_compressed(archive_path, **all_arrays)

    if save_meta:
        meta_path = os.path.join(output_dir, "excel_arrays.metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return archive_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Load Excel (.xlsx/.xls) files into NumPy arrays for ML and save as .npz with metadata."
    )
    p.add_argument("inputs", nargs="+", help="File(s) and/or directory(ies) to process recursively.")
    p.add_argument("--sheet", type=str, default=None, help="Sheet name or 0-based index. Omit for all sheets.")
    p.add_argument("--header", type=int, default=None, help="Row number (0-based) to use as header.")
    p.add_argument("--skiprows", type=int, default=0, help="Number of rows to skip at top.")
    p.add_argument("--columns", type=str, nargs="+", default=None, help="Columns by names or 0-based indices.")
    p.add_argument("--numeric-only", action="store_true", help="Keep only numeric columns.")
    p.add_argument("--dropna", action="store_true", help="Drop any row with missing values.")
    p.add_argument("--fillna", type=float, default=None, help="Fill missing values with this number.")
    p.add_argument("--scale", choices=["none", "standard", "minmax"], default="none", help="Optional scaling.")
    p.add_argument("--output-dir", type=str, default="excel_out", help="Output directory.")
    return p

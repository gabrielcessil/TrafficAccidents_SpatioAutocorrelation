import os
import subprocess
import textwrap

def keepCommonColumns(df1, df2):
    df1 = df1.copy()
    df2 = df2.copy()
    common_columns = list(set(df1.columns) & set(df2.columns))
    df1 = df1[common_columns]
    df2 = df2[common_columns]
    df1.sort_index(axis=0)
    df1.sort_index(axis=1)
    df2.sort_index(axis=0)
    df2.sort_index(axis=1)
    return df1, df2


def keepCommonRows(df1, df2, key_columns):
    # Identify common rows based on key_columns
    common_rows = df1[key_columns].merge(df2[key_columns], on=key_columns)

    # Keep only common rows **without dropping other columns**
    df1_filtered = df1[df1[key_columns].apply(tuple, axis=1).isin(common_rows.apply(tuple, axis=1))]
    df2_filtered = df2[df2[key_columns].apply(tuple, axis=1).isin(common_rows.apply(tuple, axis=1))]

    # Sort rows based on key_columns without changing column order
    df1_filtered = df1_filtered.sort_values(by=key_columns).reset_index(drop=True)
    df2_filtered = df2_filtered.sort_values(by=key_columns).reset_index(drop=True)

    # Ensure key_columns are first in both DataFrames
    df1_filtered = df1_filtered[key_columns + [col for col in df1_filtered.columns if col not in key_columns]]
    df2_filtered = df2_filtered[key_columns + [col for col in df2_filtered.columns if col not in key_columns]]

    return df1_filtered, df2_filtered
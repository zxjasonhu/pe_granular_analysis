import os
import pandas as pd

from utils.base import format_input_path


def pid_hypen_process(df, pid_hypen=False):
    if pid_hypen:
        # 3. replace "-" with "_" in PatientID
        df["PatientID"] = df["PatientID"].apply(lambda x: x.replace("_", "-"))
    else:
        df["PatientID"] = df["PatientID"].apply(lambda x: x.replace("-", "_"))

    return df


def dataframe_norm(df: pd.DataFrame, pid_hypen: bool = False) -> pd.DataFrame:
    # preprocess df:
    # 1. rename col name ANON_PatientID, ANON_StudyInstanceUID to PatientID, StudyInstanceUID
    _df = df.copy()

    if "ANON-PatientID" in _df.columns:
        _df.rename(
            columns={
                "ANON-PatientID": "PatientID",
            },
            inplace=True,
        )
    elif "ANON_PatientID" in _df.columns:
        _df.rename(
            columns={
                "ANON_PatientID": "PatientID",
            },
            inplace=True,
        )

    assert (
        "PatientID" in _df.columns
    ), f"PatientID column is missing in the DataFrame. Available columns: {list(_df.columns)}"
    _df = pid_hypen_process(_df, pid_hypen)

    # check if the columns exist
    if "StudyInstanceUID" in df.columns and "SeriesInstanceUID" in df.columns:
        return _df

    # check if the columns exist
    if "ANON-StudyUID" in _df.columns:
        _df.rename(
            columns={
                "ANON-StudyUID": "StudyInstanceUID",
            },
            inplace=True,
        )
    elif "ANON_StudyInstanceUID" in _df.columns:
        _df.rename(
            columns={
                "ANON_StudyInstanceUID": "StudyInstanceUID",
            },
            inplace=True,
        )

    if "ANON_SeriesInstanceUID" in _df.columns:
        _df.rename(
            columns={
                "ANON_SeriesInstanceUID": "SeriesInstanceUID",
            },
            inplace=True,
        )
    elif "ANON_SeriesInstanceUID" in _df.columns:
        _df.rename(
            columns={
                "ANON_SeriesInstanceUID": "SeriesInstanceUID",
            },
            inplace=True,
        )
    else:
        # 2. add SeriesInstanceUID: the last digits (split by .) of the StudyInstanceUID increment by 1
        _df["SeriesInstanceUID"] = _df["StudyInstanceUID"].apply(
            lambda x: ".".join(x.split(".")[:-1] + [str(int(x.split(".")[-1]) + 1)])
        )

    return _df


def add_path2df(df: pd.DataFrame, path: str) -> pd.DataFrame:
    df["image_folder"] = df.apply(
        lambda row: os.path.join(
            path,
            format_input_path(pid=row["PatientID"] if "PatientID" in row else None,
                              study_uid=row["StudyInstanceUID"] if "StudyInstanceUID" in row else None,
                              series_uid=row["SeriesInstanceUID"] if "SeriesInstanceUID" in row else None),
        ),
        axis=1,
    )
    # validate the path
    for index, row in df.iterrows():
        if not os.path.exists(row["image_folder"]):
            print(f"[ALERT] Path does not exist: {row['image_folder']}")

    return df

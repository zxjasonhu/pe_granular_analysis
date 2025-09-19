from typing import Optional

def determine_cam_save_name(pid: Optional[str|list], study_uid: Optional[str|list], series_uid: Optional[str|list]) -> str:
    if isinstance(pid, list):
        pid = pid[0]
    if isinstance(study_uid, list):
        study_uid = study_uid[0]
    if isinstance(series_uid, list):
        series_uid = series_uid[0]

    _name = ""
    if pid is not None and pid != "None":
        _name += f"{pid}_"
    if study_uid is not None and study_uid != "None":
        _name += f"{study_uid}_"
    if series_uid is not None and series_uid != "None":
        _name += f"{series_uid}"

    if _name.endswith("_"):
        _name = _name[:-1]
    elif _name == "":
        _name = "unknown"
    return "cam_" + _name + ".npz"
# modules/version_utils.py
import os
import re

def generate_new_version_name(existing_name: str, existing_list):
    """
    기존 파일명에서 자동으로 _v1, _v2, _v3 ... 증가시키는 함수
    예:
      train.py → train_v1.py
      train_v1.py → train_v2.py
    """

    base, ext = os.path.splitext(existing_name)

    # 이미 _v1, _v2 형태가 있는지 탐색
    m = re.search(r"_v(\d+)$", base)
    if m:
        version = int(m.group(1)) + 1
        new_base = re.sub(r"_v\d+$", f"_v{version}", base)
    else:
        # 아직 버전이 없는 경우 _v1 추가
        new_base = base + "_v1"

    new_name = new_base + ext

    # 혹시 new_name도 이미 존재한다면 재귀적으로 다시 증가
    if new_name in existing_list:
        return generate_new_version_name(new_name, existing_list)

    return new_name

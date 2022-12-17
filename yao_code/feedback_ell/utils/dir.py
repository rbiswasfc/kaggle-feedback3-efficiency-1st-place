"""
@created by: heyao
@created at: 2022-09-10 12:47:16
"""
import os


def _get_next_version(save_dir, name=None):
    name = name or "lightning_logs"
    root_dir = os.path.join(save_dir, name)
    if not os.path.isdir(root_dir):
        return 0

    existing_versions = []
    for d, *_ in os.walk(root_dir):
        bn = os.path.basename(d)
        if os.path.isdir(d) and bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace('/', '')
            existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0

    return max(existing_versions) + 1


if __name__ == '__main__':
    print(_get_next_version("/home/heyao/tf-logs"))

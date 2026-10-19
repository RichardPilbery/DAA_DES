import shutil
from pathlib import Path

root = Path(__file__).parent.parent
docs = root / "air_ambulance_des_docs"

files = {
    "README.md": "readme.qmd",
    "LICENSE": "licence.qmd",
    "CODE_OF_CONDUCT.md": "code_of_conduct.qmd",
    "checklists/STARS/STARS.qmd": "STARS.qmd",
    "checklists/STRESS-DES/STRESS_DES.qmd": "STRESS_DES.qmd",
    # "HISTORY.md": "changelog.qmd",
}

for src, dest in files.items():
    shutil.copy(root / src, docs / dest)

# YAML front matter to prepend
yaml_header = """---
format:
    html:
        toc: true
        toc-expand: 3
---

"""

# Files to prepend YAML to
for qmd_file in ["readme.qmd"]:
    file_path = docs / qmd_file
    content = file_path.read_text(encoding="utf-8")
    content = content.replace("/reference/", "../reference/")
    file_path.write_text(yaml_header + content, encoding="utf-8")

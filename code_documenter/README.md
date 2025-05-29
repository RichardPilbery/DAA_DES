This code is modified from jmlb's repository here: [https://github.com/jmlb/repoClerk/tree/main](https://github.com/jmlb/repoClerk/tree/main). All rights to the original code remain with jmlb.

It is designed to speed up the process of feeding relevant files into tools like NotebookLM as sources by placing them into fewer files.

Skipped directories, file types and specific file name patterns are controlled in config.yaml.

To run, starting in the root dir, run

`cd code_documenter`

`python repo_to_text.py --cfg ../config.yaml --repo_path ../  -w 100000 -o output_py`

Changes made by SR from original code:
- tweaks to allow exclusions based on full filepath patterns, not just filename patterns
- tweaks to fix issues with subfolders of folders in skipped dirs list not always being skipped
- tweaks to ensure that final txt output file is still written even when it does not exceed the desired file length threshold.

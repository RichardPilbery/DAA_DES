This is where all input data for the simulation should be stored.

All files in this folder - with the exception of this file and the .gitkeep file - will be ignored with regards to version control, ensuring sensitive information is not uploaded to GitHub.

**Warning:** these files are ignored by git, but they are still on disk. Do not run the AI tutorial generation tools (the Claude Code `codebase-tutorial` plugin: `/codebase-tutorial:update` or `/codebase-tutorial:add-audience`) while patient-level or other sensitive data is stored here, as their agents read the repository's working folder directly. Move the data out of the repository first. See the 'Tutorials' section of the main README.

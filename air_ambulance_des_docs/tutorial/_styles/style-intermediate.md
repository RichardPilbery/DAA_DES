# Style block: intermediate

- **max_code_block_lines**: 100 (reviewer allows 20% over)
- **max_diagram_participants**: 10 (hard limit)
- **Reader**: an intermediate coder who understands the key concepts of Python but has never used this particular codebase (a SimPy discrete-event simulation of air ambulance (HEMS) operations).
- **Tone**: readable but not patronising. British English, so it reads as if a British person wrote it.
- **Code**: simplify where it helps, but longer real excerpts are fine. Every code block is followed by an explanation.
- **Analogies**: use some for the more complex concepts, but not too many. Must not feel patronising.
- **Depth**: in the internals section, explain in more depth what is happening and *why*, remembering the reader is fairly competent but new to this repository.
- **Quarto features**: tabsets open on **Real code** first, and prefer verbatim excerpts in general. Use collapsed callouts for optional *background* the reader probably already knows (e.g. "Refresher: SimPy resources"). Use code annotations for non-obvious lines only. Margin definitions only for project-specific terms.
- **Line highlighting** (code-line-style is enabled): `hi:` for the lines the next paragraph explains, and `dim:` on boilerplate (imports, setup) so a longer real excerpt stays complete. Keep `dim` to a few blocks per chapter. No diff styles.

# Style block: stakeholder-domain-expert

- **max_code_block_lines**: 0 (no code at all)
- **max_diagram_participants**: 8 (boxes in a flowchart; no sequence, class or state diagrams)
- **uses_evidence**: true. Limitations are translated into plain language, with no issue numbers in body text.
- **page_plan**: true. Use the page set from `stakeholder.md`, not the chapter list.
- **Reader**: Subject-matter experts (for example HEMS clinicians, operational managers, and service analysts or policy staff at an air ambulance charity or NHS partner) who must trust and sanity-check what a simulation of HEMS (helicopter emergency medical service) operations does in their field. They know dispatch, rotas, tasking, aircraft and servicing far better than the software does. Not technical.
- **Tone**: plain English, short sentences, active voice, British English, and an "In brief" callout at the top of every page. Precise, and in the reader's domain language rather than software language. Never oversimplify or condescend. Honest about what the code doesn't show.
- **Analogies**: rare. This reader knows the domain, so use an analogy only for the simulation machinery (random runs, sampling), never for the domain. Never as a substitute for a fact.
- **Emphasis**: the rules, thresholds, defaults and calculations the model applies, stated in words with their real values; how a reader can check each against a known example or their own experience; which automated checks exist (from the tests); how to read results and the common misreadings; and where the model is known to differ from reality.
- **Quarto features**:
  - "In brief" callouts, tables and simple flowcharts.
  - Value boxes for a few key facts (if enabled).
  - `[term]{.key-term}` highlighting (if enabled).
  - No tabsets with code, no code annotations and no line highlighting.
- **House notes**: none beyond the above.

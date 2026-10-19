This folder contains the code relating to the Streamlit app.

This also includes all code for generating the output Quarto report.

## Quarto Output

Within the app, there is a button to generate a Quarto output.

The report template is **air_ambulance_simulation_output.qmd**, styled by **report_theme.css**. The figures it embeds, and the text blocks it reads (`quarto_text.txt`), are written by **model.py** to the `fig_outputs` folder when a simulation is run. These generated files are ignored by version control. The rendered report is saved to the `outputs` folder.

# App Pages

The pages and their navigation are defined in **app.py**.

## Control Pages

- **app.py**

## Model Setup and Running Pages
- **welcome.py** (Welcome)
- **setup.py** (Choose Model Parameters)
- **model.py** (Run Simulation)

## Model Info Pages

- **glossary.py** (Glossary of Terms)
- **info.py** (Model Details)
- **acknowledgements.py** (Acknowledgements)

**what_is.py** (Introduction to Simulation) is present but currently disabled in the navigation in app.py.

## Supporting Scripts

- **_app_utils.py**
- **_state_control.py**

## Theming and styling

- **style.css**
- **report_theme.css**

## Other folders and files

- **assets**: Images and the text content used by the pages (e.g. `glossary.csv`, `text.xlsx`), plus small helper scripts for preparing the logo images.
- **inputs**: Excel parameter templates for uploading scenarios. See [inputs/README.md](inputs/README.md).
- **fig_outputs**: Figures and text generated for the Quarto report (see above).
- **outputs**: The rendered Quarto report.
- **generate_bupar_outputs.R**: R script that uses the simulation event log to produce process maps and animations.
- **_UNUSED**: Retired scripts (scenario comparison and process animation) kept for reference.

# Visualisations

The figures shown in the app are generated when a simulation is run, and the ones used in the Quarto report are saved to `fig_outputs` (see above). The event log (`event_log.csv`) is written to the root folder of the repository.

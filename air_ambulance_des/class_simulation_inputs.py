import pandas as pd
from air_ambulance_des._processing_functions import graceful_methods


@graceful_methods
class SimulationInputs:
    def __init__(
        self, data_folder_path="../data", actual_data_folder_path="../actual_data"
    ):
        self.params_df_path = f"{data_folder_path}/run_params_used.csv"
        self.rota_path = f"{actual_data_folder_path}/HEMS_ROTA.csv"
        self.service_path = f"{data_folder_path}/service_dates.csv"
        self.callsign_path = (
            f"{actual_data_folder_path}/callsign_registration_lookup.csv"
        )
        self.rota_times = f"{actual_data_folder_path}/rota_start_end_months.csv"

        self.params_df = pd.read_csv(self.params_df_path)
        self.rota_df = pd.read_csv(self.rota_path)
        self.service_dates_df = pd.read_csv(self.service_path)
        self.callsign_registration_lookup_df = pd.read_csv(self.callsign_path)
        self.rota_start_end_months_df = pd.read_csv(self.rota_times)

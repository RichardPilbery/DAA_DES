if __name__ == "__main__":
    from air_ambulance_des.distribution_fit_utils import DistributionFitUtils

    test = DistributionFitUtils(
        "external_data/clean_daa_import_missing_2023_2024.csv",
        calculate_school_holidays=True,
    )
    # test = DistributionFitUtils('external_data/clean_daa_import.csv')
    test.import_and_wrangle()
    test.run_sim_on_historical_params()

# Testing ----------
# python distribution_fit_utils.py

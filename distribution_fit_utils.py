from csv import QUOTE_ALL
import glob
import math
import os
import sys
import numpy as np
import pandas as pd
import json
import itertools
from fitter import Fitter, get_common_distributions
from datetime import timedelta
from utils import Utils
from des_parallel_process import parallelProcessJoblib, collateRunResults, removeExistingResults
from datetime import datetime

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

class DistributionFitUtils():
    """
        # The DistributionFitUtils classa

        This class will import a CSV, undertake some light
        wrangling and then determine distributions and probabilities required
        for the Discrete Event Simulation

        example usage:
            my_data = DistributionFitUtils('data/my_data.csv')
            my_data.import_and_wrangle()

    """

    def __init__(self, file_path: str, calculate_school_holidays = False, school_holidays_years = 0):

        self.file_path = file_path
        self.df = pd.DataFrame()

        # The number of additional years of school holidays
        # that will be calculated over that maximum date in the provided dataset
        self.school_holidays_years = school_holidays_years
        self.calculate_school_holidays = calculate_school_holidays

        self.times_to_fit = [
            {"hems_result": "Patient Treated but not conveyed by HEMS",
            "times_to_fit" : ['time_allocation', 'time_mobile', 'time_to_scene', 'time_on_scene', 'time_to_clear']},
            {"hems_result": "Patient Conveyed by HEMS" , "times_to_fit" : ['time_allocation', 'time_mobile', 'time_to_scene', 'time_on_scene', 'time_to_hospital', 'time_to_clear']},
            {"hems_result": "Patient Conveyed by land with HEMS" , "times_to_fit" : ['time_allocation', 'time_mobile', 'time_to_scene', 'time_on_scene', 'time_to_hospital', 'time_to_clear']},
            {"hems_result": "Stand Down" , "times_to_fit" : ['time_allocation', 'time_mobile', 'time_to_clear']},
            {"hems_result": "Landed but no patient contact" , "times_to_fit" : ['time_allocation', 'time_mobile', 'time_to_scene', 'time_on_scene', 'time_to_clear']},
        ]

        self.sim_tools_distr_plus = [
            "poisson",
            "bernoulli",
            "triang",
            "erlang",
            "weibull_min",
            "expon_weib",
            "betabinom",
            "pearson3",
            "cauchy",
            "chi2",
            "expon",
            "exponpow",
            "gamma",
            "lognorm",
            "norm",
            "powerlaw",
            "rayleigh",
            "uniform",
            "neg_binomial",
            "zip"
        ]
        # SR 16-04-2025 Have hardcoded the common distributions
        # to make setup for random number generation more robust
        #+ get_common_distributions()

    def removeExistingResults(self, folder: str) -> None:
            """
                Removes results from previous fitting
            """

            matching_files = glob.glob(os.path.join(folder, "*.*"))

            print(matching_files)

            for file in matching_files:
                os.remove(file)


    def getBestFit(self, q_times, distr=get_common_distributions(), show_summary=False):
        """

            Convenience function for Fitter.
            Returns model and parameters that is considered
            the 'best fit'.

            TODO: Determine how Fitter works this out

        """

        if(q_times.size > 0):
            if(len(distr) > 0):
                f = Fitter(q_times, timeout=60, distributions=distr)
            else:
                f = Fitter(q_times, timeout=60)
            f.fit()
            if show_summary == True:
                f.summary()
            return f.get_best()
        else:
            return {}


    def import_and_wrangle(self):
        """

            Function to import CSV, add additional columns that are required
            and then sequentially execute other class functions to generate
            the probabilities and distributions required for the DES.

            TODO: Additional logic is required to check the imported CSV
            for missing values, incorrect columns names etc.

        """

        try:
            df = pd.read_csv(self.file_path, quoting=QUOTE_ALL)
            self.df = df

            # Perhaps run some kind of checking function here.

        except FileNotFoundError:
            print(f"Cannot locate that file")

        # If everything is okay, crack on...
        self.df['inc_date'] = pd.to_datetime(self.df['inc_date'])
        self.df['date_only'] = pd.to_datetime(df['inc_date'].dt.date)
        self.df['hour'] = self.df['inc_date'].dt.hour                      # Hour of the day
        self.df['day_of_week'] = self.df['inc_date'].dt.day_name()         # Day of the week (e.g., Monday)
        self.df['month'] = self.df['inc_date'].dt.month
        self.df['quarter'] = self.df['inc_date'].dt.quarter
        self.df['first_day_of_month'] = self.df['inc_date'].to_numpy().astype('datetime64[M]')

        # Replacing a upper quartile limit on job cycle times and
        # instead using a manually specified time frame.
        # This has the advantage of allowing manual amendment of the falues
        # on the front-end
        # self.max_values_df = self.upper_allowable_time_bounds()
        self.min_max_values_df = pd.read_csv('actual_data/upper_allowable_time_bounds.csv')
        #print(self.min_max_values_df)

        #This will be needed for other datasets, but has already been computed for DAA
        #self.df['ampds_card'] = self.df['ampds_code'].str[:2]

        self.removeExistingResults(Utils.HISTORICAL_FOLDER)
        self.removeExistingResults(Utils.DISTRIBUTION_FOLDER)


        #get proportions of AMPDS card by hour of day
        self.hour_by_ampds_card_probs()

        # Determine 'best' distributions for time-based data
        self.activity_time_distributions()

        # Calculate probability patient will be female based on AMPDS card
        self.sex_by_ampds_card_probs()

        # Determine 'best' distributions for age ranges straitifed by AMPDS card
        self.age_distributions()

        # Calculate the mean inter-arrival times stratified by yearly quarter and hour of day
        self.inter_arrival_times()

        # Alternaitve approach to IA times. Start with probabilty of call at given hour stratified by quarter
        self.hourly_arrival_by_qtr_probs()

        # Calculates the mean and standard deviaion of the number of incidents per day stratified by quarter
        self.incidents_per_day()
        self.incidents_per_day_samples()

        # Calculate probabilityy of enhanced or critical care being required based on AMPDS card
        self.enhanced_or_critical_care_by_ampds_card_probs()

        # Calculate probably of  patient outcome
        self.patient_outcome_by_care_category_and_quarter_probs()

        # Calculate HEMS result
        self.hems_reults_by_patient_outcome_and_quarter_and_vehicle_type_and_callsign_group_probs()

        # Calculate probabily of callsign being allocated to a job based on AMPDS card and hour of day
        # self.callsign_group_by_ampds_card_and_hour_probs()
        # self.callsign_group_by_ampds_card_probs()
        self.callsign_group_by_care_category()

        # Calculate probabily of HEMS result being allocated to a job based on callsign and hour of day
        self.hems_result_by_callsign_group_and_vehicle_type_probs()

        # Calculate probability of HEMS result being allocated to a job based on care category and helicopter benefit
        self.hems_result_by_care_cat_and_helicopter_benefit_probs()

        # Calculate probability of a specific patient outcome being allocated to a job based on HEMS result and callsign
        self.pt_outcome_by_hems_result_and_care_category_probs()

        # Calculate probability of a particular vehicle type based on callsign group and month of year
        self.vehicle_type_by_month_probs()
        self.vehicle_type_probs() # Similar to previous but without monthly stratification since ad hoc unavailability should account for this.

        # Calculate school holidays since servicing schedules typically avoid these dates
        if self.calculate_school_holidays:
            self.school_holidays()

        # Calculate historical data
        self.historical_monthly_totals()
        self.historical_monthly_totals_by_callsign()
        self.historical_monthly_totals_by_day_of_week()
        self.historical_median_time_of_activities_by_month_and_resource_type()
        self.historical_monthly_totals_by_hour_of_day()
        self.historical_monthly_resource_utilisation()
        self.historical_monthly_totals_all_calls()
        self.historical_daily_calls_breakdown()
        self.historical_job_durations_breakdown()
        self.historical_missed_jobs()
        self.historical_jobs_per_day_per_callsign()
        self.historical_care_cat_counts()

        # Calculate proportions of ad hoc unavailability
        self.ad_hoc_unavailability()


    def hour_by_ampds_card_probs(self):
        """

            Calculates the proportions of calls that are triaged with
            a specific AMPDS card. This is stratified by hour of day

            TODO: Determine whether this should also be stratified by yearly quarter

        """
        category_counts = self.df.groupby(['hour', 'ampds_card']).size().reset_index(name='count')
        total_counts = category_counts.groupby('hour')['count'].transform('sum')
        category_counts['proportion'] = round(category_counts['count'] / total_counts, 4)

        #category_counts['ampds_card'] = category_counts['ampds_card'].apply(lambda x: str(x).zfill(2))

        category_counts.to_csv('distribution_data/hour_by_ampds_card_probs.csv', mode="w+")


    def sex_by_ampds_card_probs(self):
        """

            Calculates the probability that the patient will be female
            stratified by AMPDS card.

        """
        age_df = self.df
        category_counts = age_df.groupby(['ampds_card', 'sex']).size().reset_index(name='count')
        total_counts = category_counts.groupby('ampds_card')['count'].transform('sum')
        category_counts['proportion'] = round(category_counts['count'] / total_counts, 3)

        category_counts[category_counts['sex'] =='Female'].to_csv('distribution_data/sex_by_ampds_card_probs.csv', mode="w+")

    def activity_time_distributions(self):
        """

            Determine the 'best' distribution for each phase of a call
            i.e. Allocation time, Mobilisation time, Time to scene
            Time on scene, Travel time to hospital and handover, Time to clear.
            Not all times will apply to all cases, so the class 'times_to_fit'
            variable is a list of dictionaries, which contains the times to fit

            The data is currently stratitied by HEMS_result and vehicle type fields.

        """

        vehicle_type = self.df['vehicle_type'].dropna().unique()

        # We'll need to make sure that where a distribution is missing that the time is set to 0 in the model.
        # Probably easier than complicated logic to determine what times should be available based on hems_result

        final_distr = []

        for row in self.times_to_fit:
            #print(row)
            for ttf in row['times_to_fit']:
                for vt in vehicle_type:
                    #print(f"HEMS result is {row['hems_result']} times_to_fit is {ttf} and vehicle type is  {vt}")

                    # This line might not be required if data quality is determined when importing the data
                    max_time = self.min_max_values_df[self.min_max_values_df['time'] == ttf].max_value_mins.iloc[0]
                    min_time = self.min_max_values_df[self.min_max_values_df['time'] == ttf].min_value_mins.iloc[0]

                    #print(f"Max time is {max_time} and Min time is {min_time}")

                    if ttf == 'time_on_scene':
                        # There is virtually no data for HEMS_result other than patient conveyed
                        # which is causing issues with fitting. For time on scene, will
                        # use a simplified fitting ignoring hems_result as a category
                        fit_times = self.df[
                            (self.df.vehicle_type == vt) &
                            (self.df[ttf] >= min_time) &
                            (self.df[ttf] <= max_time)
                        ][ttf]
                    else:
                        fit_times = self.df[
                            (self.df.vehicle_type == vt) &
                            (self.df[ttf] >= min_time) &
                            (self.df[ttf] <= max_time) &
                            (self.df.hems_result == row['hems_result'])
                        ][ttf]
                    #print(fit_times[:10])
                    best_fit = self.getBestFit(fit_times, distr=self.sim_tools_distr_plus)
                    #print(best_fit)

                    return_dict = { "vehicle_type": vt, "time_type" : ttf, "best_fit": best_fit, "hems_result": row['hems_result'], "n": len(fit_times)}
                    #print(return_dict)
                    final_distr.append(return_dict)

        with open('distribution_data/activity_time_distributions.txt', 'w+') as convert_file:
            convert_file.write(json.dumps(final_distr))
        convert_file.close()


    def age_distributions(self):
        """

            Determine the 'best' distribution for age stratified by
            AMPDS card

        """

        age_distr = []

        age_df = self.df[["age", "ampds_card"]].dropna()
        ampds_cards = age_df['ampds_card'].unique()
        print(ampds_cards)

        for card in ampds_cards:
            fit_ages = age_df[age_df['ampds_card'] == card]['age']
            best_fit = self.getBestFit(fit_ages, distr=self.sim_tools_distr_plus)
            return_dict = { "ampds_card": str(card), "best_fit": best_fit, "n": len(fit_ages)}
            age_distr.append(return_dict)

        with open('distribution_data/age_distributions.txt', 'w+') as convert_file:
            convert_file.write(json.dumps(age_distr))
        convert_file.close()


    def inter_arrival_times(self):
        """

            Calculate the mean inter-arrival times for patients
            stratified by hour, and and yearly quarter

        """

        ia_df = self.df[['date_only', 'quarter', 'hour']].dropna()

        count_df = ia_df.groupby(['hour', 'date_only', 'quarter']).size().reset_index(name='n')

        ia_times_df = (
            count_df.groupby(['hour', 'quarter'])
            .agg(
                # max_arrivals_per_hour=('n', lambda x: round(60 / np.max(x), 3)),
                # min_arrivals_per_hour=('n', lambda x: round(60 / np.min(x),3)),
                mean_cases=('n', lambda x: round(x.mean(), 1)),
                # sd_cases=('n', lambda x: round(x.std(), 3)),
                mean_iat=('n', lambda x: 60 / x.mean())
                # n=('n', 'size')
            )
            .reset_index()
        )
        # Additional column for NSPPThinning
        ia_times_df['t'] = ia_times_df['hour']
        ia_times_df['arrival_rate'] = ia_times_df['mean_iat'].apply(lambda x: 1/x)

        ia_times_df.to_csv('distribution_data/inter_arrival_times.csv', mode='w+')


    def incidents_per_day(self):
        """
        Fit distributions for number of incidents per day using actual daily counts,
        applying year-based weighting to reflect trends (e.g., 2024 busier than 2023),
        stratified by season and quarter.
        """
        import math
        import json
        import numpy as np

        inc_df = self.df[['inc_date', 'date_only', 'quarter']].copy()
        inc_df['year'] = inc_df['date_only'].dt.year

        # Daily incident counts
        inc_per_day = inc_df.groupby('date_only').size().reset_index(name='jobs_per_day')
        inc_per_day['year'] = inc_per_day['date_only'].dt.year

        # Merge quarter and season from self.df
        date_info = self.df[['date_only', 'quarter']].drop_duplicates()

        if 'season' not in self.df.columns:
            date_info['season'] = date_info['quarter'].map(lambda q: "winter" if q in [1, 4] else "summer")
        else:
            date_info = date_info.merge(
                self.df[['date_only', 'season']].drop_duplicates(),
                on='date_only',
                how='left'
            )

        inc_per_day = inc_per_day.merge(date_info, on='date_only', how='left')

        # Weight settings - simple implementation rather than biased mean thing
        year_weights = {
            2023: 1.0,
            2024: 4.0  # 10% more weight to 2024
        }

        # ========== SEASONAL DISTRIBUTIONS ==========
        jpd_distr = []

        for season in inc_per_day['season'].dropna().unique():
            filtered = inc_per_day[inc_per_day['season'] == season].copy()
            filtered['weight'] = filtered['year'].map(year_weights).fillna(1.0)

            # Repeat rows proportionally by weight
            replicated = filtered.loc[
                filtered.index.repeat((filtered['weight'] * 10).round().astype(int))
            ]['jobs_per_day']

            best_fit = self.getBestFit(np.array(replicated), distr=self.sim_tools_distr_plus)

            jpd_distr.append({
                "season": season,
                "best_fit": best_fit,
                "min_n_per_day": int(replicated.min()),
                "max_n_per_day": int(replicated.max()),
                "mean_n_per_day": float(replicated.mean())
            })

        with open('distribution_data/inc_per_day_distributions.txt', 'w+') as f:
            json.dump(jpd_distr, f)

        # ========== QUARTERLY DISTRIBUTIONS ==========
        jpd_qtr_distr = []

        for quarter in inc_per_day['quarter'].dropna().unique():
            filtered = inc_per_day[inc_per_day['quarter'] == quarter].copy()
            filtered['weight'] = filtered['year'].map(year_weights).fillna(1.0)

            replicated = filtered.loc[
                filtered.index.repeat((filtered['weight'] * 10).round().astype(int))
            ]['jobs_per_day']

            best_fit = self.getBestFit(np.array(replicated), distr=self.sim_tools_distr_plus)

            jpd_qtr_distr.append({
                "quarter": int(quarter),
                "best_fit": best_fit,
                "min_n_per_day": int(replicated.min()),
                "max_n_per_day": int(replicated.max()),
                "mean_n_per_day": float(replicated.mean())
            })

        with open('distribution_data/inc_per_day_qtr_distributions.txt', 'w+') as f:
            json.dump(jpd_qtr_distr, f)

    def incidents_per_day_samples(self, weight_map=None, scale_factor=10):
        """
            Create weighted empirical samples of incidents per day by season and quarter.
        """

        inc_df = self.df[['date_only', 'quarter']].copy()
        inc_df['year'] = inc_df['date_only'].dt.year
        inc_df['season'] = inc_df['quarter'].map(lambda q: "winter" if q in [1, 4] else "summer")

        # Get raw counts per day
        daily_counts = inc_df.groupby('date_only').size().reset_index(name='jobs_per_day')
        daily_counts['year'] = daily_counts['date_only'].dt.year

        # Merge back in season/quarter info
        meta_info = self.df[['date_only', 'quarter']].drop_duplicates()
        if 'season' in self.df.columns:
            meta_info = meta_info.merge(
                self.df[['date_only', 'season']].drop_duplicates(),
                on='date_only', how='left'
            )
        else:
            meta_info['season'] = meta_info['quarter'].map(lambda q: "winter" if q in [1, 4] else "summer")

        daily_counts = daily_counts.merge(meta_info, on='date_only', how='left')

        # Year weight map
        if weight_map is None:
            weight_map = {2023: 1.0, 2024: 1.1}

        # Compute weights
        daily_counts['weight'] = daily_counts['year'].map(weight_map).fillna(1.0)

        # Storage
        empirical_samples = {}

        # Season-based
        for season in daily_counts['season'].dropna().unique():
            filtered = daily_counts[daily_counts['season'] == season].copy()
            repeated = filtered.loc[
                filtered.index.repeat((filtered['weight'] * scale_factor).round().astype(int))
            ]['jobs_per_day'].tolist()

            empirical_samples[season] = repeated

        # Quarter-based
        for quarter in daily_counts['quarter'].dropna().unique():
            filtered = daily_counts[daily_counts['quarter'] == quarter].copy()
            repeated = filtered.loc[
                filtered.index.repeat((filtered['weight'] * scale_factor).round().astype(int))
            ]['jobs_per_day'].tolist()

            empirical_samples[f"Q{int(quarter)}"] = repeated

        with open("distribution_data/inc_per_day_samples.json", 'w') as f:
            json.dump(empirical_samples, f)


    def enhanced_or_critical_care_by_ampds_card_probs(self):
        """

            Calculates the probabilty of enhanced or critica care resource beign required
            based on the AMPDS card

        """

        ec_df = self.df[['ampds_card', 'ec_benefit', 'cc_benefit']].copy()

        def assign_care_category(row):
            # There are some columns with both EC and CC benefit selected
            # this function will allocate to only 1
            if row['cc_benefit'] == 'y':
                return 'CC'
            elif row['ec_benefit'] == 'y':
                return 'EC'
            else:
                return 'REG'

        ec_df['care_category'] = ec_df.apply(assign_care_category, axis = 1)

        care_cat_counts = ec_df.groupby(['ampds_card', 'care_category']).size().reset_index(name='count')
        total_counts = care_cat_counts.groupby('ampds_card')['count'].transform('sum')

        care_cat_counts['proportion'] = round(care_cat_counts['count'] / total_counts, 3)

        care_cat_counts.to_csv('distribution_data/enhanced_or_critical_care_by_ampds_card_probs.csv', mode = "w+", index = False)

    def patient_outcome_by_care_category_and_quarter_probs(self):
        """

            Calculates the probabilty of a patient outcome based on care category and yearly quarter

        """

        po_df = self.df[['quarter', 'ec_benefit', 'cc_benefit', 'pt_outcome']].copy()

        def assign_care_category(row):
            # There are some columns with both EC and CC benefit selected
            # this function will allocate to only 1
            if row['cc_benefit'] == 'y':
                return 'CC'
            elif row['ec_benefit'] == 'y':
                return 'EC'
            else:
                return 'REG'
            
        # There are some values that are missing e.g. CC quarter 1 Deceased
        # I think we've had problems when trying to sample from this kind of thing before
        # As a fallback, ensure that 'missing' combinations are given a count and proportion of 0
        outcomes = po_df['pt_outcome'].unique()
        care_categories = ['CC', 'EC', 'REG']
        quarters = po_df['quarter'].unique()

        all_combinations = pd.DataFrame(list(itertools.product(outcomes, care_categories, quarters)),
                                    columns=['pt_outcome', 'care_category', 'quarter'])

        po_df['care_category'] = po_df.apply(assign_care_category, axis = 1)

        po_cat_counts = po_df.groupby(['pt_outcome', 'care_category', 'quarter']).size().reset_index(name='count')

        merged = pd.merge(all_combinations, po_cat_counts, 
                      on=['pt_outcome', 'care_category', 'quarter'], 
                      how='left').fillna({'count': 0})
        merged['count'] = merged['count'].astype(int)

        total_counts = merged.groupby(['care_category', 'quarter'])['count'].transform('sum')
        merged['proportion'] = round(merged['count'] / total_counts.replace(0, 1), 3) 

        merged.to_csv('distribution_data/patient_outcome_by_care_category_and_quarter_probs.csv', mode = "w+", index = False)

    def hems_reults_by_patient_outcome_and_quarter_and_vehicle_type_and_callsign_group_probs(self):
        """

            Calculates the probabilty of a given HEMS result based on 
            patient outcome, yearly quarter, vehicle type and callsign group

        """

        hr_df = self.df[['quarter', 'pt_outcome', 'vehicle_type', 'callsign_group']].copy()
            
        # There are some values that are missing e.g. CC quarter 1 Deceased
        # I think we've had problems when trying to sample from this kind of thing before
        # As a fallback, ensure that 'missing' combinations are given a count and proportion of 0
        outcomes = hr_df['pt_outcome'].unique()
        vehicle_categories = hr_df['vehicle_type'].unique()
        callsign_group_categories = hr_df['callsign_group'].unique()
        quarters = hr_df['quarter'].unique()

        all_combinations = pd.DataFrame(list(itertools.product(outcomes, vehicle_categories, callsign_group_categories, quarters)),
                                    columns=['pt_outcome', 'vehicle_type', 'callsign_group', 'quarter'])

        hr_cat_counts = hr_df.groupby(['pt_outcome', 'vehicle_type', 'callsign_group', 'quarter']).size().reset_index(name='count')

        merged = pd.merge(all_combinations, hr_cat_counts, 
                      on=['pt_outcome', 'vehicle_type', 'callsign_group', 'quarter'], 
                      how='left').fillna({'count': 0})
        merged['count'] = merged['count'].astype(int)

        total_counts = merged.groupby(['vehicle_type', 'callsign_group', 'quarter'])['count'].transform('sum')
        merged['proportion'] = round(merged['count'] / total_counts.replace(0, 1), 3) 

        merged.to_csv('distribution_data/hems_reults_by_patient_outcome_and_quarter_and_vehicle_type_and_callsign_group_probs.csv', mode = "w+", index = False)



    def hourly_arrival_by_qtr_probs(self):
        """

            Calculates the proportions of calls arriving in any given hour
            stratified by yearly quarter

        """

        ia_df = self.df[['quarter', 'hour']].dropna()

        hourly_counts = ia_df.groupby(['hour', 'quarter']).size().reset_index(name='count')
        total_counts = hourly_counts.groupby(['quarter'])['count'].transform('sum')
        hourly_counts['proportion'] = round(hourly_counts['count'] / total_counts, 4)

        hourly_counts.sort_values(by=['quarter', 'hour']).to_csv('distribution_data/hourly_arrival_by_qtr_probs.csv', mode="w+")


    def callsign_group_by_ampds_card_and_hour_probs(self):
        """

            Calculates the probabilty of a specific callsign being allocated to
            a call based on the AMPDS card category and hour of day

        """
        callsign_counts = self.df.groupby(['ampds_card', 'hour', 'callsign_group']).size().reset_index(name='count')

        total_counts = callsign_counts.groupby(['ampds_card', 'hour'])['count'].transform('sum')
        callsign_counts['proportion'] = round(callsign_counts['count'] / total_counts, 4)

        callsign_counts.to_csv('distribution_data/callsign_group_by_ampds_card_and_hour_probs.csv', mode = "w+", index=False)

    def callsign_group_by_ampds_card_probs(self):
        """

            Calculates the probabilty of a specific callsign being allocated to
            a call based on the AMPDS card category

        """

        callsign_df = self.df[self.df['callsign_group'] != 'Other']

        callsign_counts = callsign_df.groupby(['ampds_card', 'callsign_group']).size().reset_index(name='count')

        total_counts = callsign_counts.groupby(['ampds_card'])['count'].transform('sum')
        callsign_counts['proportion'] = round(callsign_counts['count'] / total_counts, 4)

        callsign_counts.to_csv('distribution_data/callsign_group_by_ampds_card_probs.csv', mode = "w+", index=False)

    def callsign_group_by_care_category(self):
        """

            Calculates the probabilty of a specific callsign being allocated to
            a call based on the care category

        """
        callsign_df = (
            self.df
            .assign(
                helicopter_benefit=np.select(
                    [
                        self.df["cc_benefit"] == "y",
                        self.df["ec_benefit"] == "y",
                        self.df["hems_result"].isin([
                            "Stand Down En Route",
                            "Landed but no patient contact",
                            "Stand Down Before Mobile"
                        ])
                    ],
                    ["y", "y", "n"],
                    default=self.df["helicopter_benefit"]
                ),
                care_category=np.select(
                    [
                        self.df["cc_benefit"] == "y",
                        self.df["ec_benefit"] == "y"
                    ],
                    ["CC", "EC"],
                    default="REG"
                )
            )
        )

        callsign_df = callsign_df[callsign_df['callsign_group'] != 'Other']

        callsign_counts = callsign_df.groupby(['care_category', 'callsign_group']).size().reset_index(name='count')

        total_counts = callsign_counts.groupby(['care_category'])['count'].transform('sum')
        callsign_counts['proportion'] = round(callsign_counts['count'] / total_counts, 4)

        callsign_counts.to_csv('distribution_data/callsign_group_by_care_category_probs.csv', mode = "w+", index=False)

    def vehicle_type_by_month_probs(self):
        """

            Calculates the probabilty of a car/helicopter being allocated to
            a call based on the callsign group and month of the year

        """
        callsign_counts = self.df.groupby(['callsign_group', 'month', 'vehicle_type']).size().reset_index(name='count')

        total_counts = callsign_counts.groupby(['callsign_group', 'month'])['count'].transform('sum')
        callsign_counts['proportion'] = round(callsign_counts['count'] / total_counts, 4)

        callsign_counts.to_csv('distribution_data/vehicle_type_by_month_probs.csv', mode = "w+")

    def vehicle_type_probs(self):
        """

            Calculates the probabilty of a car/helicopter being allocated to
            a call based on the callsign group

        """

        callsign_counts = self.df.groupby(['callsign_group', 'vehicle_type']).size().reset_index(name='count')

        total_counts = callsign_counts.groupby(['callsign_group'])['count'].transform('sum')
        callsign_counts['proportion'] = round(callsign_counts['count'] / total_counts, 4)

        callsign_counts.to_csv('distribution_data/vehicle_type_probs.csv', mode = "w+")


    def hems_result_by_callsign_group_and_vehicle_type_probs(self):
        """

            Calculates the probabilty of a specific HEMS result being allocated to
            a call based on the callsign group and hour of day

            TODO: These probability calculation functions could probably be refactored into a single
            function and just specify columns and output name

        """
        hems_counts = self.df.groupby(['hems_result', 'callsign_group', 'vehicle_type']).size().reset_index(name='count')

        total_counts = hems_counts.groupby(['callsign_group', 'vehicle_type'])['count'].transform('sum')
        hems_counts['proportion'] = round(hems_counts['count'] / total_counts, 4)

        hems_counts.to_csv('distribution_data/hems_result_by_callsign_group_and_vehicle_type_probs.csv', mode = "w+", index=False)


    def hems_result_by_care_cat_and_helicopter_benefit_probs(self):
        """

            Calculates the probabilty of a specific HEMS result being allocated to
            a call based on the care category amd whether a helicopter is beneficial

        """

        # Wrangle the data...trying numpy for a change

        hems_df = (
            self.df
            .assign(
                helicopter_benefit=np.select(
                    [
                        self.df["cc_benefit"] == "y",
                        self.df["ec_benefit"] == "y",
                        self.df["hems_result"].isin([
                            "Stand Down En Route",
                            "Landed but no patient contact",
                            "Stand Down Before Mobile"
                        ])
                    ],
                    ["y", "y", "n"],
                    default=self.df["helicopter_benefit"]
                ),
                care_cat=np.select(
                    [
                        self.df["cc_benefit"] == "y",
                        self.df["ec_benefit"] == "y"
                    ],
                    ["CC", "EC"],
                    default="REG"
                )
            )
        )

        hems_counts = hems_df.groupby(['hems_result', 'care_cat', 'helicopter_benefit']).size().reset_index(name='count')

        hems_counts['total'] = hems_counts.groupby(['care_cat', 'helicopter_benefit'])['count'].transform('sum')
        hems_counts['proportion'] = round(hems_counts['count'] / hems_counts['total'], 4)

        hems_counts.to_csv('distribution_data/hems_result_by_care_cat_and_helicopter_benefit_probs.csv', mode = "w+", index=False)


    def pt_outcome_by_hems_result_and_care_category_probs(self):
        """

            Calculates the probabilty of a specific patient outcome based on HEMS result

        """

        hems_df = (
            self.df
            .assign(
                helicopter_benefit=np.select(
                    [
                        self.df["cc_benefit"] == "y",
                        self.df["ec_benefit"] == "y",
                        self.df["hems_result"].isin([
                            "Stand Down En Route",
                            "Landed but no patient contact",
                            "Stand Down Before Mobile"
                        ])
                    ],
                    ["y", "y", "n"],
                    default=self.df["helicopter_benefit"]
                ),
                care_category=np.select(
                    [
                        self.df["cc_benefit"] == "y",
                        self.df["ec_benefit"] == "y"
                    ],
                    ["CC", "EC"],
                    default="REG"
                )
            )
        )

        po_counts = hems_df.groupby(['pt_outcome', 'hems_result', 'care_category']).size().reset_index(name='count')

        po_counts['total'] = po_counts.groupby(['hems_result', 'care_category'])['count'].transform('sum')
        po_counts['proportion'] = round(po_counts['count'] / po_counts['total'], 4)

        po_counts.to_csv('distribution_data/pt_outcome_by_hems_result_and_care_category_probs.csv', mode = "w+")


    def school_holidays(self) -> None:
        """"
            Function to generate a CSV file containing schoole holiday
            start and end dates for a given year. The Year range is determined
            by the submitted data (plus a year at the end of the study for good measure)
        """

        min_date = self.df.inc_date.min()
        max_date = self.df.inc_date.max() + timedelta(weeks = (52 * self.school_holidays_years))

        u = Utils()

        years_of_holidays_list = u.years_between(min_date, max_date)

        sh = pd.DataFrame(columns=['year', 'start_date', 'end_date'])

        for i, year in enumerate(years_of_holidays_list):
            tmp = u.calculate_term_holidays(year)

            if i == 0:
                sh = tmp
            else:
                sh = pd.concat([sh, tmp])

        sh.to_csv('actual_data/school_holidays.csv', index = False)


# These functions are to wrangle historical data to provide comparison against the simulation outputs

    def historical_jobs_per_day_per_callsign(self):
        df = self.df

        df["date"] = pd.to_datetime(df["inc_date"]).dt.date
        all_counts_hist = df.groupby(["date", "callsign"])["job_id"].count().reset_index()
        all_counts_hist.rename(columns={'job_id':'jobs_in_day'}, inplace=True)

        all_combinations = pd.DataFrame(
            list(itertools.product(df['date'].unique(), df['callsign'].unique())),
            columns=['date', 'callsign']
        ).dropna()

        merged = all_combinations.merge(all_counts_hist, on=['date', 'callsign'], how='left')
        merged['jobs_in_day'] = merged['jobs_in_day'].fillna(0).astype(int)

        all_counts = merged.groupby(['callsign', 'jobs_in_day']).count().reset_index().rename(columns={"date":"count"})
        all_counts.to_csv("historical_data/historical_jobs_per_day_per_callsign.csv", index=False)

    def historical_care_cat_counts(self):
        df_historical = self.df

        df_historical['inc_date'] = pd.to_datetime(df_historical['inc_date'])

        df_historical['month_start'] = df_historical.inc_date.dt.strftime("%Y-%m-01")
        df_historical['hour'] = df_historical.inc_date.dt.hour

        conditions = [
            df_historical['cc_benefit'] == 'y',
            df_historical['ec_benefit'] == 'y',
            df_historical['helicopter_benefit'] == 'y',
            df_historical['callsign_group'] == 'Other'
        ]

        choices = [
            'CC',
            'EC',
            'REG - helicopter benefit',
            'Unknown - DAA resource did not attend'
        ]

        df_historical['care_category'] = np.select(conditions, choices, default='REG')

        historical_value_counts_by_hour = (
            df_historical.value_counts(["hour", "care_category"])
            .reset_index(name="count")
            )

        (historical_value_counts_by_hour
         .sort_values(['hour', 'care_category'])
         .to_csv("historical_data/historical_care_cat_counts.csv"))

        # Also output the % of regular (not cc/ec) jobs with a helicopter benefit

        numerator = historical_value_counts_by_hour[historical_value_counts_by_hour["care_category"] == "REG - helicopter benefit"]["count"].sum()

        denominator = historical_value_counts_by_hour[historical_value_counts_by_hour["care_category"] != "Unknown - DAA resource did not attend"]["count"].sum()

        with open('distribution_data/proportion_jobs_heli_benefit.txt', 'w+') as heli_benefit_file:
            heli_benefit_file.write(json.dumps((numerator/denominator).round(4)))
        heli_benefit_file.close()


    def historical_monthly_totals(self):
        """
            Calculates monthly incident totals from provided dataset of historical data
        """

        # Multiple resources can be sent to the same job.
        monthly_df = self.df[['inc_date', 'first_day_of_month', 'hems_result', 'vehicle_type']]\
            .drop_duplicates(subset="inc_date", keep="first")

        is_stand_down = monthly_df['hems_result'].str.contains("Stand Down")
        monthly_df['stand_down_car'] = ((monthly_df['vehicle_type'] == "car") & is_stand_down).astype(int)
        monthly_df['stand_down_helicopter'] = ((monthly_df['vehicle_type'] == "helicopter") & is_stand_down).astype(int)

        monthly_totals_df = monthly_df.groupby('first_day_of_month').agg(
                                stand_down_car=('stand_down_car', 'sum'),
                                stand_down_helicopter=('stand_down_helicopter', 'sum'),
                                total_jobs=('vehicle_type', 'size')
                            ).reset_index()

        monthly_totals_df.rename(columns={'first_day_of_month': 'month'}).to_csv('historical_data/historical_jobs_per_month.csv', mode="w+", index=False)

    def historical_monthly_totals_all_calls(self):
        """
            Calculates monthly incident totals from provided dataset of historical data stratified by callsign
        """

        # Multiple resources can be sent to the same job.
        monthly_df = self.df[['inc_date', 'first_day_of_month']].dropna()

        monthly_totals_df = monthly_df.groupby(['first_day_of_month']).count().reset_index()

        monthly_totals_df.rename(columns={'first_day_of_month': 'month'}).to_csv('historical_data/historical_monthly_totals_all_calls.csv', mode="w+", index=False)

    def historical_monthly_totals_by_callsign(self):
        """
            Calculates monthly incident totals from provided dataset of historical data stratified by callsign
        """

        # Multiple resources can be sent to the same job.
        monthly_df = self.df[['inc_date', 'first_day_of_month', 'callsign']].dropna()

        monthly_totals_df = monthly_df.groupby(['first_day_of_month', 'callsign']).count().reset_index()

        #print(monthly_totals_df.head())

        monthly_totals_pivot_df = monthly_totals_df.pivot(index='first_day_of_month', columns='callsign', values='inc_date').fillna(0).reset_index().rename_axis(None, axis=1)

        #print(monthly_totals_pivot_df.head())

        monthly_totals_pivot_df.rename(columns={'first_day_of_month': 'month'}).to_csv('historical_data/historical_monthly_totals_by_callsign.csv', mode="w+", index=False)

    def historical_monthly_totals_by_hour_of_day(self):
        """
            Calculates monthly incident totals from provided dataset of historical data stratified by hour of the day
        """

        # Multiple resources can be sent to the same job.
        monthly_df = self.df[['inc_date', 'first_day_of_month', 'hour']].dropna()\
            .drop_duplicates(subset="inc_date", keep="first")

        monthly_totals_df = monthly_df.groupby(['first_day_of_month', 'hour']).count().reset_index()

        #print(monthly_totals_df.head())

        monthly_totals_pivot_df = monthly_totals_df.pivot(index='first_day_of_month', columns='hour', values='inc_date').fillna(0).reset_index().rename_axis(None, axis=1)

        #print(monthly_totals_pivot_df.head())

        monthly_totals_pivot_df.rename(columns={'first_day_of_month': 'month'}).to_csv('historical_data/historical_monthly_totals_by_hour_of_day.csv', mode="w+", index=False)

    def historical_monthly_totals_by_day_of_week(self):
        """
            Calculates number of incidents per month stratified by day of the week
        """

        # Multiple resources can be sent to the same job.
        monthly_df = self.df[['inc_date', 'first_day_of_month', 'day_of_week']].dropna()\
            .drop_duplicates(subset="inc_date", keep="first")

        monthly_totals_df = monthly_df.groupby(['first_day_of_month', 'day_of_week']).count().reset_index()

        #print(monthly_totals_df.head())

        monthly_totals_pivot_df = monthly_totals_df.pivot(index='first_day_of_month', columns='day_of_week', values='inc_date').fillna(0).reset_index().rename_axis(None, axis=1)

        #print(monthly_totals_pivot_df.head())

        monthly_totals_pivot_df.rename(columns={'first_day_of_month': 'month'}).to_csv('historical_data/historical_monthly_totals_by_day_of_week.csv', mode="w+", index=False)

    def historical_median_time_of_activities_by_month_and_resource_type(self):
        """
            Calculate the median time for each of the job cycle phases stratified by month and vehicle type
        """

        median_df = self.df[['first_day_of_month', 'time_allocation',
                             'time_mobile', 'time_to_scene', 'time_on_scene',
                             'time_to_hospital', 'time_to_clear', 'vehicle_type']]

        median_df['total_job_time'] = median_df[[
            'time_allocation', 'time_mobile', 'time_to_scene', 'time_on_scene',
            'time_to_hospital', 'time_to_clear']].sum(axis=1)

        # Replacing zeros with NaN to exclude from median calculation
        # since if an HEMS result is Stood down en route, then time_on_scene would be zero and affect the median
        median_df.replace(0, np.nan, inplace=True)

        # Grouping by month and resource_type, calculating medians
        median_times = median_df.groupby(['first_day_of_month', 'vehicle_type']).median(numeric_only=True).reset_index()

        pivot_data = median_times.pivot_table(
            index='first_day_of_month',
            columns='vehicle_type',
            values=['time_allocation', 'time_mobile', 'time_to_scene', 'time_on_scene', 'time_to_hospital', 'time_to_clear', 'total_job_time']
        )

        pivot_data.columns = [f"median_{col[1]}_{col[0]}" for col in pivot_data.columns]
        pivot_data = pivot_data.reset_index()

        pivot_data.rename(columns={'first_day_of_month': 'month'}).to_csv('historical_data/historical_median_time_of_activities_by_month_and_resource_type.csv', mode="w+", index=False)

    def historical_monthly_resource_utilisation(self):
        """
            Calculates number of, and time spent on, incidents per month stratified by callsign
        """

        # Multiple resources can be sent to the same job.
        monthly_df = self.df[[
            'inc_date', 'first_day_of_month', 'callsign', 'time_allocation',
            'time_mobile', 'time_to_scene', 'time_on_scene', 'time_to_hospital',
            'time_to_clear']]

        monthly_df['total_time'] = monthly_df.filter(regex=r'^time_').sum(axis=1)

        monthly_totals_df = monthly_df.groupby(['callsign', 'first_day_of_month'], as_index=False)\
            .agg(n = ('callsign', 'size'), total_time = ('total_time', 'sum'))

        monthly_totals_pivot_df = monthly_totals_df.pivot(index='first_day_of_month', columns='callsign', values=['n', 'total_time'])

        monthly_totals_pivot_df.columns = [f"{col[0]}_{col[1]}" for col in  monthly_totals_pivot_df.columns]
        monthly_totals_pivot_df = monthly_totals_pivot_df.reset_index()

        monthly_totals_pivot_df.rename(columns={'first_day_of_month': 'month'}).to_csv('historical_data/historical_monthly_resource_utilisation.csv', mode="w+", index=False)

    def historical_daily_calls_breakdown(self):

        df = self.df
        # Convert inc_date to date only (remove time)
        df['date'] = pd.to_datetime(df['inc_date']).dt.date

        # Count number of calls per day
        calls_in_day_breakdown = df.groupby('date').size().reset_index(name='calls_in_day')

        # Save the daily call counts with a 'day' index column
        calls_in_day_breakdown_with_day = calls_in_day_breakdown.copy()
        calls_in_day_breakdown_with_day.insert(0, 'day', range(1, len(calls_in_day_breakdown) + 1))
        calls_in_day_breakdown_with_day.drop(columns='date').to_csv('historical_data/historical_daily_calls_breakdown.csv', index=False)

        # Count how many days had the same number of calls
        calls_per_day_summary = calls_in_day_breakdown['calls_in_day'].value_counts().reset_index()
        calls_per_day_summary.columns = ['calls_in_day', 'days']
        calls_per_day_summary.to_csv('historical_data/historical_daily_calls.csv', index=False)

    def historical_missed_jobs(self):
        df = self.df
        df["date"] = pd.to_datetime(df["inc_date"])
        df["hour"] = df["date"].dt.hour
        df["month_start"] = df["date"].dt.strftime("%Y-%m-01")
        df["callsign_group_simplified"] = df["callsign_group"].apply(lambda x: "No HEMS available" if x=="Other" else "HEMS (helo or car) available and sent")
        df["quarter"] = df["inc_date"].dt.quarter

        # By month
        count_df_month = df[["callsign_group_simplified", "month_start"]].value_counts().reset_index(name="count").sort_values(['callsign_group_simplified','month_start'])
        count_df_month.to_csv("historical_data/historical_missed_calls_by_month.csv", index=False)

        # By hour
        count_df = df[["callsign_group_simplified", "hour"]].value_counts().reset_index(name="count").sort_values(['callsign_group_simplified','hour'])
        count_df.to_csv("historical_data/historical_missed_calls_by_hour.csv", index=False)

        # By quarter and hour
        count_df_quarter = df[["callsign_group_simplified", "quarter", "hour"]].value_counts().reset_index(name="count").sort_values(['quarter','callsign_group_simplified','hour'])
        count_df_quarter.to_csv("historical_data/historical_missed_calls_by_quarter_and_hour.csv", index=False)

    def upper_allowable_time_bounds(self):
        """
            Calculates the maximum permissable time for each phase on an incident based on supplied historical data.
            This is currently set to 1.5x the upper quartile of the data distribution
        """

        median_df = self.df[['time_allocation', 'time_mobile', 'time_to_scene', 'time_on_scene', 'time_to_hospital', 'time_to_clear', 'vehicle_type']].dropna()

        # Replacing zeros with NaN to exclude from median calculation
        # since if an HEMS result is Stood down en route, then time_on_scene would be zero and affect the median
        median_df.replace(0, np.nan, inplace=True)

        print(median_df.quantile(.75))
        # pivot_data.rename(columns={'first_day_of_month': 'month'}).to_csv('historical_data/historical_median_time_of_activities_by_month_and_resource_type.csv', mode="w+", index=False)

    def historical_job_durations_breakdown(self):

        df = self.df

        cols = [
            'callsign', 'vehicle_type',
            'time_allocation', 'time_mobile',
            'time_to_scene', 'time_on_scene',
            'time_to_hospital', 'time_to_clear'
        ]
        df2 = df[cols].copy()

        # 2. Add a 1-based row identifier
        df2['job_identifier'] = range(1, len(df2) + 1)

        # 3. Compute total_duration as the row-wise sum of the time columns
        time_cols = [
            'time_allocation', 'time_mobile',
            'time_to_scene', 'time_on_scene',
            'time_to_hospital', 'time_to_clear'
        ]
        df2['total_duration'] = df2[time_cols].sum(axis=1, skipna=True)

        #print(df2.head())

        # 4. Pivot (melt) to long format
        df_long = df2.melt(
            id_vars=['job_identifier', 'callsign', 'vehicle_type'],
            value_vars=time_cols + ['total_duration'],
            var_name='name',
            value_name='value'
        )

        #print(df_long[df_long.job_identifier == 1])

        # 5. Drop any rows where callsign or vehicle_type is missing
        df_long = df_long.dropna(subset=['callsign', 'vehicle_type'])
        df_long_sorted = df_long.sort_values("job_identifier").reset_index(drop=True)

        # 6. Write out to CSV
        df_long_sorted.to_csv("historical_data/historical_job_durations_breakdown.csv", index=False)



    def calculate_availability_row(self, row, rota_df, callsign_lookup_df):
        """
            Compute downtime overlap, rota-based scheduled time, and proportion for a given row.
            Returns data tagged with bin, quarter, and downtime reason.
        """

        registration = row['aircraft'].lower()
        downtime_start = pd.to_datetime(row['offline'], utc=True)
        downtime_end = pd.to_datetime(row['online'], utc=True)
        reason = row.get('reason', None)

        hour = downtime_start.hour
        if 0 <= hour <= 5:
            six_hour_bin = '00-05'
        elif 6 <= hour <= 11:
            six_hour_bin = '06-11'
        elif 12 <= hour <= 17:
            six_hour_bin = '12-17'
        else:
            six_hour_bin = '18-23'

        quarter = downtime_start.quarter

        # Match callsign
        match = callsign_lookup_df[callsign_lookup_df['registration'].str.lower() == registration]
        if match.empty:
            return {
                'registration': registration,
                'offline': downtime_start,
                'online': downtime_end,
                'six_hour_bin': six_hour_bin,
                'quarter': quarter,
                'total_offline': None,
                'scheduled_minutes': None,
                'reason': reason,
                'proportion': None
            }
        callsign = match.iloc[0]['callsign']

        rota_rows = rota_df[rota_df['callsign'] == callsign]
        if rota_rows.empty:
            return {
                'registration': registration,
                'offline': downtime_start,
                'online': downtime_end,
                'six_hour_bin': six_hour_bin,
                'quarter': quarter,
                'total_offline': None,
                'scheduled_minutes': None,
                'reason': reason,
                'proportion': None
            }

        month = downtime_start.month
        season = 'summer' if month in [4, 5, 6, 7, 8, 9] else 'winter'

        total_scheduled_minutes = 0
        total_overlap_minutes = 0

        for _, rota in rota_rows.iterrows():
            start_hour = rota[f'{season}_start']
            end_hour = rota[f'{season}_end']

            for base_day in [downtime_start.normalize() - timedelta(days=1),
                            downtime_start.normalize(),
                            downtime_start.normalize() + timedelta(days=1)]:

                rota_start = base_day + timedelta(hours=start_hour)
                rota_end = base_day + timedelta(hours=end_hour)
                if end_hour <= start_hour:
                    rota_end += timedelta(days=1)

                overlap_start = max(downtime_start, rota_start)
                overlap_end = min(downtime_end, rota_end)

                if overlap_end > overlap_start:
                    scheduled_minutes = (rota_end - rota_start).total_seconds() / 60
                    overlap_minutes = (overlap_end - overlap_start).total_seconds() / 60

                    total_scheduled_minutes += scheduled_minutes
                    total_overlap_minutes += overlap_minutes

        if total_scheduled_minutes == 0:
            proportion = None
        else:
            proportion = total_overlap_minutes / total_scheduled_minutes

        return {
            'registration': registration,
            'offline': downtime_start,
            'online': downtime_end,
            'six_hour_bin': six_hour_bin,
            'quarter': quarter,
            'total_offline': total_overlap_minutes,
            'scheduled_minutes': total_scheduled_minutes,
            'reason': reason,
            'proportion': proportion
        }

    def ad_hoc_unavailability(self):
        """
        Process ad hoc unavailability records into a stratified probability table
        based on six-hour bin and quarterly calendar period.
        """

        adhoc_df = pd.read_csv('external_data/ad_hoc.csv', parse_dates=['offline', 'online'])
        adhoc_df = adhoc_df[['aircraft', 'offline', 'online', 'reason']]

        rota_df = pd.read_csv("actual_data/HEMS_ROTA.csv")
        callsign_lookup_df = pd.read_csv("actual_data/callsign_registration_lookup.csv")

        results = adhoc_df.apply(
            lambda row: self.calculate_availability_row(row, rota_df, callsign_lookup_df),
            axis=1
        )
        final_df = pd.DataFrame(results.tolist())

        print(final_df)

        # downtime by bin + quarter + reason
        grouped = final_df.groupby(['registration', 'six_hour_bin', 'quarter', 'reason'])['total_offline'].sum().reset_index()

        # Scheduled time by bin + quarter (for calculating availability)
        scheduled_totals = final_df.groupby(['registration','six_hour_bin', 'quarter'])['scheduled_minutes'].sum().reset_index()
        scheduled_totals = scheduled_totals.rename(columns={'scheduled_minutes': 'total_scheduled'})

        # Merge downtime total
        downtime_totals = grouped.groupby(['registration','six_hour_bin', 'quarter'])['total_offline'].sum().reset_index()
        downtime_totals = downtime_totals.rename(columns={'total_offline': 'total_downtime'})

        # Calculate availability
        availability_df = pd.merge(scheduled_totals, downtime_totals, on=['registration', 'six_hour_bin', 'quarter'], how='left').fillna(0)
        availability_df['reason'] = 'available'
        availability_df['probability'] = (availability_df['total_scheduled'] - availability_df['total_downtime']) / availability_df['total_scheduled']
        available_probs = availability_df[['registration', 'six_hour_bin', 'quarter', 'reason', 'probability']]

        # Add unavailability reason probabilities
        # Merge rota time into grouped
        grouped = pd.merge(grouped, scheduled_totals, on=['registration', 'six_hour_bin', 'quarter'], how='left')
        grouped['probability'] = grouped['total_offline'] / grouped['total_scheduled']

        # Combine everything
        final_prob_df = pd.concat([
            grouped[['registration', 'six_hour_bin', 'quarter', 'reason', 'probability']],
            available_probs
        ], ignore_index=True)

        final_prob_df = final_prob_df.sort_values(by=['registration', 'quarter', 'six_hour_bin', 'reason']).reset_index(drop=True)
        final_prob_df.to_csv("distribution_data/ad_hoc_unavailability.csv", index=False)


    def run_sim_on_historical_params(self):
        # Ensure all rotas are using default values
        rota = pd.read_csv("tests/rotas_historic/HISTORIC_HEMS_ROTA.csv")
        rota.to_csv("actual_data/HEMS_ROTA.csv", index=False)

        callsign_reg_lookup = pd.read_csv("tests/rotas_historic/HISTORIC_callsign_registration_lookup.csv")
        callsign_reg_lookup.to_csv("actual_data/callsign_registration_lookup.csv", index=False)

        service_history = pd.read_csv("tests/rotas_historic/HISTORIC_service_history.csv")
        service_history.to_csv("actual_data/service_history.csv", index=False)

        service_sched = pd.read_csv("tests/rotas_historic/HISTORIC_service_schedules_by_model.csv")
        service_sched.to_csv("actual_data/service_schedules_by_model.csv", index=False)

        print("Generating simulation results...")
        removeExistingResults()

        total_runs = 12
        sim_duration = 60 * 24 * 7 * 52 * 2, # 2 years

        parallelProcessJoblib(
            total_runs=total_runs,
            sim_duration=sim_duration,
            warm_up_time=0,
            sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
            amb_data=False,
            print_debug_messages=True
        )

        collateRunResults()

    results_all_runs = pd.read_csv("data/run_results.csv")
    # Also run the model to get some base-case outputs
    resource_requests = results_all_runs[results_all_runs["event_type"] == "resource_request_outcome"].copy()
    resource_requests["care_cat"] = resource_requests.apply(lambda x: "REG - Helicopter Benefit" if x["heli_benefit"]=="y" and x["care_cat"]=="REG" else x["care_cat"], axis=1)
    missed_jobs_care_cat_summary = resource_requests[["care_cat", "time_type"]].value_counts().reset_index(name="jobs").sort_values(["care_cat", "time_type"]).copy()
    missed_jobs_care_cat_summary["jobs_average"] = (missed_jobs_care_cat_summary["jobs"]/12)
    missed_jobs_care_cat_summary["jobs_per_year_average"] = (missed_jobs_care_cat_summary["jobs_average"]/730*365).round(0)

    missed_jobs_care_cat_summary.to_csv("historical_data/calculated/SIM_hist_params_missed_jobs_care_cat_summary.csv")

if __name__ == "__main__":
    from distribution_fit_utils import DistributionFitUtils
    test = DistributionFitUtils('external_data/clean_daa_import_missing_2023_2024.csv', True)
    #test = DistributionFitUtils('external_data/clean_daa_import.csv')
    test.import_and_wrangle()
    test.run_sim_on_historical_params()

# Testing ----------
# python distribution_fit_utils.py

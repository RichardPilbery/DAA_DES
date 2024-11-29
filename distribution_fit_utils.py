import numpy as np
import pandas as pd
import json
from fitter import Fitter, get_common_distributions

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

    def __init__(self, file_path: str):
       
        self.file_path = file_path
        self.df = pd.DataFrame()

        self.times_to_fit = [
            {"hems_result": "Patient Treated (not conveyed)", 
            "times_to_fit" : ['time_allocation', 'time_mobile', 'time_to_scene', 'time_on_scene', 'time_to_clear'], "pt_outcome": ['Deceased', 'Conveyed by land without DAA', 'Unknown']},
            {"hems_result": "Patient Conveyed" , "times_to_fit" : ['time_allocation', 'time_mobile', 'time_to_scene', 'time_on_scene', 'time_to_hospital', 'time_to_clear'], "pt_outcome" : ['Conveyed by land with DAA', 'Airlifted']},
            {"hems_result": "Stand Down Before Mobile" , "times_to_fit" : ['time_allocation', 'time_to_clear'],"pt_outcome" : ['Unknown']},
            {"hems_result": "Stand Down En Route" , "times_to_fit" : ['time_allocation', 'time_mobile', 'time_to_clear'] , "pt_outcome" : ['Unknown']},
            {"hems_result": "Landed but no patient contact" , "times_to_fit" : ['time_allocation', 'time_mobile', 'time_to_scene', 'time_on_scene', 'time_to_clear'], "pt_outcome" : ['Unknown']},
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
        ] + get_common_distributions()

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

        #print(f"Open file {self.file_path}")
        try:
            df = pd.read_csv(self.file_path)
            self.df = df
            
            # Perhaps run some kind of checking function here.

        except FileNotFoundError:
            print(f"Cannot locate that file")

        # If everything is okay, crack on...
        self.df['inc_date'] = pd.to_datetime(self.df['inc_date'])
        self.df['date_only'] = pd.to_datetime(df['inc_date'].dt.date)
        self.df['hour'] = self.df['inc_date'].dt.hour                      # Hour of the day
        self.df['day_of_week'] = self.df['inc_date'].dt.day_name()         # Day of the week (e.g., Monday)
        self.df['quarter'] = self.df['inc_date'].dt.quarter   

        # This will be needed for other datasets, but has already been computed for DAA
        # self.df['ampds_card'] = self.df['ampds_code'].str[:2]

        # get proportions of AMPDS card by hour of day
        self.hour_by_ampds_card_probs()

        # Determine 'best' distributions for time-based data
        self.activity_time_distributions()

        # Calculate probability patient will be female based on AMPDS card
        self.sex_by_ampds_card_probs()

        # Determine 'best' distributions for age ranges straitifed by AMPDS card
        self.age_distributions()

        # Calculate the mean inter-arrival times stratified by yearly quarter and hour of day
        self.inter_arrival_times()
            

    def hour_by_ampds_card_probs(self):
        """
        
            Calculates the proportions of calls that are triaged with 
            a specific AMPDS card. This is stratified by hour of day

            TODO: Determine whether this should also be stratified by yearly quarter
        
        """
        category_counts = self.df.groupby(['hour', 'ampds_card']).size().reset_index(name='count')
        total_counts = category_counts.groupby('hour')['count'].transform('sum')
        category_counts['proportion'] = round(category_counts['count'] / total_counts, 4)

        with open('distribution_data/hour_by_ampds_card_probs.csv', 'w+') as convert_file:
            convert_file.write(category_counts.to_csv())
        convert_file.close()


    def sex_by_ampds_card_probs(self):
        """
        
            Calculates the probability that the patient will be female
            stratified by AMPDS card.
        
        """

        age_df = self.df
        category_counts = age_df.groupby(['ampds_card', 'sex']).size().reset_index(name='count')
        total_counts = category_counts.groupby('ampds_card')['count'].transform('sum')
        category_counts['proportion'] = round(category_counts['count'] / total_counts, 3)

        with open('distribution_data/sex_by_ampds_card_probs.csv', 'w+') as convert_file:
            convert_file.write(category_counts[category_counts['sex'] =='Female'].to_csv())
        convert_file.close()


    def activity_time_distributions(self):
        """
        
            Determine the 'best' distribution for each phase of a call
            i.e. Allocation time, Mobilisation time, Time to scene
            Time on scene, Travel time to hospital and handover, Time to clear.
            Not all times will apply to all cases, so the class 'times_to_fit'
            variable is a list of dictionaries, which contains the times to fit
            
            The data is currently stratitied by HEMS callsign, HEMS_result and pt_outcome fields.
        
        """
       
        hems_cs = self.df['callsign'].unique()

        # We'll need to make sure that where a distribution is missing that the time is set to 0 in the model.
        # Probably easier than complicated logic to determine what times should be available based on hems_result

        final_distr = []

        for row in self.times_to_fit:
            print(row)
            for ttf in row['times_to_fit']:
                for cs in hems_cs:
                    for pto in row['pt_outcome']:
                        #print(f"HEMS result is {row['hems_result']} cs is {cs} and times_to_fit is {ttf} and patient outcome {pto}")

                        # This line might not be required if data quality is determined when importing the data
                        max_time = 20 if ttf == "time_mobile" else 120
                        fit_times = self.df[
                            (self.df.callsign == cs) & 
                            (self.df[ttf] > 0) & 
                            (self.df[ttf] < max_time) & 
                            (self.df.hems_result == row['hems_result']) & 
                            (self.df.pt_outcome == pto)
                        ][ttf]
                        #print(fit_times[:10])
                        best_fit = self.getBestFit(fit_times, distr=self.sim_tools_distr_plus)
                        return_dict = { "callsign": cs, "time_type" : ttf, "best_fit": best_fit, "hems_result": row['hems_result'], "pt_outcome": pto, "n": len(fit_times)}
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

        for card in ampds_cards:
            fit_ages = age_df[age_df['ampds_card'] == card]['age']
            best_fit = self.getBestFit(fit_ages, distr=self.sim_tools_distr_plus)
            # Note that wrapping card in int() prevents JSON whinging about not supporting 64 bit floats...
            return_dict = { "ampds_card": int(card), "best_fit": best_fit, "n": len(fit_ages)}
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
                max_arrival_rate=('n', lambda x: round(60 / np.max(x), 3)),
                min_arrival_rate=('n', lambda x: round(60 / np.min(x),3)),
                mean_cases=('n', lambda x: round(x.mean(), 3)),
                sd_cases=('n', lambda x: round(x.std(), 3)), 
                mean_inter_arrival_time=('n', lambda x: round(60 / x.mean(),3)),
                n=('n', 'size')
            )
            .reset_index()
        )

        with open('distribution_data/inter_arrival_times.csv', 'w+') as convert_file:
            convert_file.write(ia_times_df.to_csv())
        convert_file.close()


# Testing ----------
# python
# from distribution_fit_utils import DistributionFitUtils
# test = DistributionFitUtils('external_data/clean_daa_import.csv')
# test.import_and_wrangle()
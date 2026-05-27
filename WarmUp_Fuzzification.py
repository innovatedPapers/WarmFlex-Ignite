import numpy as np
from skfuzzy import gaussmf, trimf
from skfuzzy import control as ctrl
import pandas as pd



# Calculate fuzzy parameters of input
def calculate_fuzzy_parameters(input_file):
    # Calculate percentiles for the splits; 33rd and 67th percentiles are just examples
    low_high_threshold = np.percentile(input_file, 33)
    medium_high_threshold = np.percentile(input_file, 67)

    # Split the array based on the calculated thresholds
    low_range = input_file[input_file <= low_high_threshold]
    medium_range = input_file[(input_file > low_high_threshold) & (input_file <= medium_high_threshold)]
    high_range = input_file[input_file > medium_high_threshold]

    low_mean, low_std = np.mean(low_range), np.std(low_range)
    medium_mean, medium_std = np.mean(medium_range), np.std(medium_range)
    high_mean, high_std = np.mean(high_range), np.std(high_range)

    # Create a dictionary to store the fuzzy parameters
    fuzzy_parameters = {
        'low_mean': low_mean,
        'low_std': low_std,
        'medium_mean': medium_mean,
        'medium_std': medium_std,
        'high_mean': high_mean,
        'high_std': high_std
    }

    return fuzzy_parameters


def function_member(func):
    if func in k0_data:
        return 0
    elif func in k1_data:
        return 1
    elif func in k2_data:
        return 2


def setup_fuzzy_system(input_file):
    # creating inter_arrival range (unchanged)
    inter_arrival_range = np.linspace(0, 1440, input_file)
    inter_arrival_times = ctrl.Antecedent(inter_arrival_range, 'inter_arrival_times')
    result_fuzzy_parameters = calculate_fuzzy_parameters(inter_arrival_range)
    inter_arrival_times['low'] = gaussmf(inter_arrival_range, mean=result_fuzzy_parameters['low_mean'], sigma=result_fuzzy_parameters['low_std'])
    inter_arrival_times['medium'] = gaussmf(inter_arrival_range, mean=result_fuzzy_parameters['medium_mean'], sigma=result_fuzzy_parameters['medium_std'])
    inter_arrival_times['high'] = gaussmf(inter_arrival_range, mean=result_fuzzy_parameters['high_mean'], sigma=result_fuzzy_parameters['high_std'])

    # creating label range (unchanged)
    label_antecedent = ctrl.Antecedent(np.arange(0, 3), 'label')
    label_antecedent['0'] = trimf(label_antecedent.universe, [0, 0, 0.5])
    label_antecedent['1'] = trimf(label_antecedent.universe, [1, 1, 1.5])
    label_antecedent['2'] = trimf(label_antecedent.universe, [2, 2, 2.5])

    # >>> OUTPUT RANGE: 0 to 3 minutes (fine resolution) <<<
    warm_up_range = np.arange(0.0, 3.0 + 1e-9, 0.01)  # 0.00 .. 3.00 minutes
    result_output_parameters = calculate_fuzzy_parameters(warm_up_range)
    warm_up_period = ctrl.Consequent(warm_up_range, 'warm_up_period')
    warm_up_period['low']    = gaussmf(warm_up_period.universe, mean=result_output_parameters['low_mean'],    sigma=result_output_parameters['low_std'])
    warm_up_period['medium'] = gaussmf(warm_up_period.universe, mean=result_output_parameters['medium_mean'], sigma=result_output_parameters['medium_std'])
    warm_up_period['high']   = gaussmf(warm_up_period.universe, mean=result_output_parameters['high_mean'],   sigma=result_output_parameters['high_std'])


    # Rules (unchanged)
    rule1 = ctrl.Rule((inter_arrival_times['low'] | label_antecedent['2']), warm_up_period['high'])
    rule2 = ctrl.Rule((inter_arrival_times['medium'] | label_antecedent['1']), warm_up_period['medium'])
    rule3 = ctrl.Rule((inter_arrival_times['high'] | label_antecedent['0']), warm_up_period['low'])

    warm_up_period_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    warm_up_period_sim = ctrl.ControlSystemSimulation(warm_up_period_ctrl)
    return inter_arrival_times, warm_up_period_ctrl, warm_up_period_sim




def fuzzy_conclusion(function_name, inter_arrival_time):
    warm_up_period_sim.input['inter_arrival_times'] = inter_arrival_time
    warm_up_period_sim.input['label'] = function_member(function_name)
    warm_up_period_sim.compute()
    # Ensure result is strictly within [0, 5] minutes
    return float(np.clip(warm_up_period_sim.output['warm_up_period'], 0.0, 3.0))


unique_funcs_addr = "/proj/p4topo-PG0/myCodes/project/simulator/"
# number of unique functions
input_file = len(np.sort(pd.read_csv(unique_funcs_addr+"inter_arrival_time.csv", low_memory=False, usecols=["mean inter-arrival time"]).to_numpy().ravel()))

# importing clusters
cluster_path = "/proj/p4topo-PG0/myCodes/project/clustering/silhouette_clustering/pca/data/"
k0_data = set(pd.read_csv(cluster_path + "result_kmeans_k0_euclidean.csv", usecols=["HashFunction"]).values.ravel())
k1_data = set(pd.read_csv(cluster_path + "result_kmeans_k1_euclidean.csv", usecols=["HashFunction"]).values.ravel())
k2_data = set(pd.read_csv(cluster_path + "result_kmeans_k2_euclidean.csv", usecols=["HashFunction"]).values.ravel())


# getting value of inter_arrival_times and warm_up_period_ctrl
inter_arrival_times, warm_up_period_ctrl, warm_up_period_sim = setup_fuzzy_system(input_file)





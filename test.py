from FID_integration import import_data, FID_integration
from FID_General import plot_chromatogram, plot_chromatogram_cluster, load_results, delete_samples
from bouqueter import clusterer

# %% Automatic with clustering
# %%%Plot and check retention times/figures
# # '/Users/gerard/Documents/GitHub/RAW_records/Bolshoye Shchuchye/Waxes/Raw FID Data/BolShoye Files'
data, *_ = import_data()
plot_chromatogram(data, '/Users/gerard/Documents/GitHub/RAW_records/Bolshoye Shchuchye/Waxes/Raw FID Data/BolShoye Files/HF/A/chromatograms', time_window=[4,30])
# %%% Re
clusterer(data, '/Users/gerard/Documents/GitHub/RAW_records/Bolshoye Shchuchye/Waxes/Raw FID Data/BolShoye Files/HF/A/chromatograms', max_clusters=2)
plot_chromatogram_cluster(data)
data = FID_integration(categorized=data, gaussian_fit_mode='single')


# %% Automatic
data = FID_integration(gaussian_fit_mode='single', smoothing_window=5)
# %% Manual
data = FID.integration(gaussian_fit_mode='single', manual_peak_integration=True, smoothing_window=13, peak_labels=True)


# %%
test = load_results('/Users/gerard/Documents/GitHub/RAW_records/Bolshoye Shchuchye/Waxes/Raw FID Data/BolShoye Files/HF/A/chromatoPy output/FID_output.json')
for key in test['Samples'].keys():
    print(key)
# %%  
fp = '/Users/gerard/Documents/GitHub/RAW_records/Bolshoye Shchuchye/Waxes/Raw FID Data/BolShoye Files/HF/A/chromatoPy output/FID_output.json'
FID.delete_samples(fp, ['HF041', 'HF045', 'HF049', 'HF050', 'HF051'])
# %%


# '/Users/gerard/Documents/GitHub/RAW_records/Bolshoye Shchuchye/Waxes/Raw FID Data/BolShoye Files/HF/C'

# %%
import chromatopy
from chromatopy import FID

# data = FID.integration(gaussian_fit_mode='single', smoothing_window=5)
fp = '/Users/gerard/Documents/GitHub/RAW_records/Bolshoye Shchuchye/Waxes/Raw FID Data/BolShoye Files/HF/C/chromatoPy output/FID_output.json'
FID.delete_samples(fp, ['HF027', 'HF029', 'HF031', 'HF032'])

# %%

# NOW RUN THE FILES AGAIN IN MANUAL MODE - TEST for why it isnt working!!!

# %%
from FID_General import plot_chromatogram, plot_chromatogram_cluster, load_results, delete_samples
fp = '/Users/gerard/Documents/GitHub/RAW_records/Bolshoye Shchuchye/Waxes/Raw FID Data/BolShoye Files/HF/C/chromatoPy output/FID_output.json'
dat = load_results(fp)
for key in dat['Samples'].keys():
    print(key)
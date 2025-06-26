import chromatopy
from chromatopy import FID

# 1.
# data = FID.plot_chromatogram(time_window=[5,30])
# %%
FID.clusterer(data, '/Users/gerard/Documents/GitHub/RAW_records/Bolshoye Shchuchye/Waxes/Raw FID Data/BolShoye Files/OC/chromatograms', max_clusters=4)
# %%
FID.plot_chromatogram_cluster(data)
# %%
data = FID.integration(categorized=data, gaussian_fit_mode='single')
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
FID.integration(categorized = data, gaussian_fit_mode='single')
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
    
# %%
FID.integration(manual_peak_integration=True, peak_labels=True)

# %%
fp = '/Users/gerard/Documents/GitHub/RAW_records/Bolshoye Shchuchye/Waxes/Raw FID Data/BolShoye Files/HF/B/chromatoPy output/FID_output.json'
data = FID.load_results(fp)
FID.delete_samples(fp, ['HF035', 'HF036'])

# %%w
import chromatopy
from chromatopy import FID
fp = '/Users/gerard/Documents/GitHub/RAW_records/Bolshoye Shchuchye/Waxes/Raw FID Data/BolShoye Files/HF/C/chromatoPy output'
data = FID.load_results(fp)
samps = []
for x in data['Samples'].keys():
    print(x)
# fp = '/Users/gerard/Documents/GitHub/RAW_records/Bolshoye Shchuchye/Waxes/Raw FID Data/BolShoye Files/HF/C/chromatoPy output/FID_output.json'
# FID.delete_samples(fp, ['HF032', 'HF033'])
    
# %%
for key in data['Samples'].keys():
    if 'Processed Data' not in data['Samples'][key]:
        print(key)
# %%
for key in data['Samples'].keys():
    if 'Processed Data' not in data["Samples"][key].keys():
        print(f"Processed data not in {key}")
        continue
    if 'C24' not in data['Samples'][key]['Processed Data'].keys():
        print(f"C24 not in {key}")
        
# %%

data = FID.load_results('/Users/gerard/Documents/GitHub/RAW_records/Bolshoye Shchuchye/Waxes/Raw FID Data/BolShoye Files/OC/chromatoPy output')
for key in data['Samples'].keys():
    if 'Processed Data' not in data['Samples'][key].keys():
        print(key)
        
        
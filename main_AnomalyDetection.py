import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import TimeSeriesTransformation as tst
import Plotter
import Analysis
import Utils
# UTILITIES
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def make_table(df, filename):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    # Set Times New Roman font
    font_properties = fm.FontProperties(family='Times New Roman', size=12)

    # Create table
    the_table = ax.table(cellText=df.values,
                         colLabels=df.columns,
                         loc='center',
                         cellLoc='center')

    # Apply font to all cells and headers
    for key, cell in the_table.get_celld().items():
        cell.get_text().set_fontproperties(font_properties)
        cell.get_text().set_horizontalalignment('center')
        cell.get_text().set_verticalalignment('center')

    # Auto scale column widths
    the_table.auto_set_column_width(col=list(range(len(df.columns))))

    # Save as PDF
    with PdfPages(f"{filename}.pdf") as pp:
        pp.savefig(fig, bbox_inches='tight')



# ANOMALY ANALYSIS
def determine_phenomenon(slope):
    if float(slope) > 1: Phenomenon = 'Intensification' # Above the y=x
    elif float(slope) > 0 and float(slope) < 1: Phenomenon = 'Attenuation' # Below the  y=x
    else: Phenomenon = 'Flip' # Downwards curve
    return Phenomenon
    
def determine_score_type(value):
    if float(value) > 0: score = 'Above the average' 
    else: score = 'Below the average'
    return score
    
def determine_lmi_type(value):
    """Determines the LMI score based on the value."""
    if float(value) > 0:
        return 'Same profiles'
    else:
        return 'Opposite profiles'


# PROCESS DATAFRAMES

def process_LMI_anomalies(LMI_lin_anom, LMI_PreLockdown, scores_PreLockdown):

    if LMI_lin_anom.empty:
        print("No LMI findings to be processed.")
        return pd.DataFrame()

    # Conside only rows with identified anomalies
    score_anomalies = LMI_lin_anom[LMI_lin_anom['Anomaly'] != '']

    # Initialize empty dataframe
    score_anomalies_perState = pd.DataFrame(columns = ['Anomaly', 'Attribute', 'Class', 'Profile Before Lockdown', 'Phenomenon after Lockdown'])
    
    
    for index, row in score_anomalies.iterrows():
     
        attribute = row['Attribute']
        class_ = row['Class']
        
        # Itens can be separated by commas
        slope = row['Anomaly Slope']
        state = row['Anomaly']
        
        
        lmi = LMI_PreLockdown[(LMI_PreLockdown['Attribute'] == attribute) & (LMI_PreLockdown['Class'] == class_)][state].values[0]
        score = scores_PreLockdown[(scores_PreLockdown['Attribute'] == attribute) & (scores_PreLockdown['Class'] == class_)][state].values[0]     



        # Use info to classify anomaly 
        Phenomenon = determine_phenomenon(slope)                
        score_type = determine_score_type(score)
        lmi_type = determine_lmi_type(lmi)

        # Create columns values
        new_row = {'Anomaly': state,
                   'Attribute': attribute,
                   'Class': class_, 
                   'Profile Before Lockdown': score_type,
                   'Neighborhood Profile before Lockdown': lmi_type,
                   'Phenomenon after Lockdown': Phenomenon}
        
        # Add row
        score_anomalies_perState.loc[len(score_anomalies_perState)] = new_row
            
    return score_anomalies_perState
            
            
def process_Diss_Anomalies(scr_lin_anom, scores_PreLockdown):

    if scr_lin_anom.empty:
        print("No dissimilarity findings to be processed.")
        return pd.DataFrame()

    # Conside only rows with identified anomalies
    score_anomalies = scr_lin_anom[scr_lin_anom['Anomaly'] != '']

    # Initialize empty dataframe
    score_anomalies_perState = pd.DataFrame(columns = ['Anomaly', 'Attribute', 'Class', 'Profile Before Lockdown', 'Phenomenon after Lockdown'])
    
    
    for index, row in score_anomalies.iterrows():
     
        attribute = row['Attribute']
        class_ = row['Class']
        
        # Itens can be separated by commas
        slope = row['Anomaly Slope']
        state = row['Anomaly']
        
                
        score = scores_PreLockdown[(scores_PreLockdown['Attribute'] == attribute) & (scores_PreLockdown['Class'] == class_)][state]
        score = score.values[0]
        # Use info to classify anomaly 
        Phenomenon = determine_phenomenon(slope)                
        score_type = determine_score_type(score)
        
        # Create columns values
        new_row = {'Anomaly': state,
                   'Attribute': attribute,
                   'Class': class_, 
                   'Profile Before Lockdown': score_type,
                   'Phenomenon after Lockdown': Phenomenon}
        
        # Add row
        score_anomalies_perState.loc[len(score_anomalies_perState)] = new_row
            
    return score_anomalies_perState


        

            
            

csv_file_name = r'PreProc_Data.csv'
shp_file_name = r'BR_UF_2022/BR_UF_2022.shp'
granularity = 1
count = 'months'
alpha = 2

# Create base dataframe
df_entire = pd.read_csv(csv_file_name,low_memory=False, parse_dates=['data_inversa'])
df = tst.filter_dates_between(df_entire, '01 Mar 2018', '28 Feb 2022', date_column='data_inversa')


# SHOWS THE MAJOR INFLUENCES ON TOTAL ACCIDENTS AFTER LOCKDOWN
# Time series of total amount of occurences (not splitted between classes)
totalTimeSeries_daily = tst.create_time_series(df)
Plotter.PlotTimeSeries(
    totalTimeSeries_daily,
    count='days',
    granularity=7,
    vline_dates=['16 Mar 2020', '16 Mar 2019'],
    vline_labels=["Lockdown\nDate", 'One Year\nBefore\nLockdown'],
    title='TotalAccidents_TimeSeries/Total_')



# COMPUTE SCORES ON BOTH PERIODS
df_PreLockdown = tst.filter_dates_between(df, '16 Mar 2020', '16 Aug 2020', date_column='data_inversa')
df_Lockdown = tst.filter_dates_between(df, '16 Mar 2019', '16 Aug 2019', date_column='data_inversa')
# Perform the spatial correlation score analysis
scores_PreLockdown, LMI_PreLockdown, GMI_PreLockdown, PL_Correlations = Analysis.SpatilAnalysis(
    df_PreLockdown, 
    granularity=7, 
    count='days', 
    plotNeighborhood=False,  
    plotMoran=False, 
    plotMaps=False,
    title="Spatial_Analysis/PreLockdown",
    shp_file_name=shp_file_name)
# Perform the dissimilarity score analysis
scores_Lockdown, LMI_Lockdown, GMI_Lockdown, L_Correlations = Analysis.SpatilAnalysis(
    df_Lockdown,
    granularity=7,
    count='days',
    plotNeighborhood=False,
    plotMoran=False,
    plotMaps=False,
    title="Spatial_Analysis/Lockdown",
    shp_file_name=shp_file_name)




# DISSIMILARITY ANALYSIS
print("Dissimilarity Analysis starting...")
# Get columns present in both to be compared
common_scores_PreLockdown , common_scores_Lockdown = Utils.keepCommonColumns(scores_PreLockdown, scores_Lockdown)
common_scores_PreLockdown,  common_scores_Lockdown = Utils.keepCommonRows(common_scores_PreLockdown, common_scores_Lockdown, key_columns=['Attribute', 'Class'])
# Register info
make_table(common_scores_PreLockdown, "Tables/common_scores_PreLockdown")
make_table(common_scores_Lockdown, "Tables/common_scores_Lockdown")

# Analyze the profile changes
scr_regr, scr_lin_anom = Analysis.ProfileChangeAnalysis(
    common_scores_PreLockdown,
    common_scores_Lockdown, 
    filename="Spatial_Analysis/DisimilarityScores/")

# Register info
make_table(scr_regr, "Tables/scores_Regression_findings")
make_table(scr_lin_anom, "Tables/scores_Anomalous_findings")


# Classify anomalies
score_anomalies_perState = process_Diss_Anomalies(scr_lin_anom, scores_PreLockdown)
# Register info
make_table(score_anomalies_perState, "Tables/Score_anomalies")

criteria = scr_regr[(scr_regr['Acceptable Temporal Linearity'] == True) & (scr_regr['Acceptable Residuals Normality'] == True)]
pairs = criteria[['Attribute', 'Class']]
result = scr_lin_anom.merge(pairs, on=['Attribute', 'Class'], how='inner')
make_table(result, "Tables/Score_anomalies_Acceptable")

# Plot scores highliting annomalies
#Plotter.PlotComparison(scr_regr, scr_lin_anom, common_scores_PreLockdown, common_scores_Lockdown, title="Spatial_Analysis/DisimilarityScores/", annomaliesOnly=False)





# SPATIAL-TEMPORAL AUTOCORRELATION ANALYSIS
print("Spatio-Temporal Analysis starting...")
# Get columns present in both to be compared
common_LMI_PreLockdown , common_LMI_Lockdown = Utils.keepCommonColumns(LMI_PreLockdown, LMI_Lockdown)
common_LMI_PreLockdown, common_LMI_Lockdown = Utils.keepCommonRows(common_LMI_PreLockdown, common_LMI_Lockdown, key_columns=['Attribute', 'Class'])
# Analyze the profile changes
LMI_regr, LMI_lin_anom = Analysis.ProfileChangeAnalysis(
    common_LMI_PreLockdown,
    common_LMI_Lockdown,
    filename="Spatial_Analysis/LMI/")
# Register info
make_table(LMI_lin_anom, "Tables/LMI_Anomalous_findings")
make_table(LMI_regr, "Tables/LMI_Regression_findings")

# Classify anomalies
LMI_anomalies_perState = process_LMI_anomalies(LMI_lin_anom, LMI_PreLockdown, scores_PreLockdown)
make_table(LMI_anomalies_perState, "Tables/LMI_anomalies_perState")

# Plot scores highliting annomalies
#Plotter.PlotComparison(LMI_regr, LMI_lin_anom, common_LMI_PreLockdown, common_LMI_Lockdown,title="Spatial_Analysis/LMI/", annomaliesOnly=False)



criteria = LMI_regr[(LMI_regr['Acceptable Temporal Linearity'] == True) & (LMI_regr['Acceptable Residuals Normality'] == True)]
pairs = criteria[['Attribute', 'Class']]
result = LMI_lin_anom.merge(pairs, on=['Attribute', 'Class'], how='inner')
make_table(result, "Tables/LMI_anomalies_perState_Acceptable")


import TimeSeriesAutoCorrelation as tsac
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import Dictionaries
import TimeSeriesTransformation as tst
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from adjustText import adjust_text
import matplotlib.patheffects as pe
from scipy import stats
import Plotter as plt


def keep_rows(df, attribute, values): return df[df[attribute].isin(values)]

def Validate(globalIndex, localIndexes, gamma=0.05):
    E = - 1/ (len(localIndexes)-1)
    teste = globalIndex < E*(1+gamma) or E*(1-gamma) < globalIndex
    return teste

def PlotHeatmap(scores,title):
    scores_df = pd.DataFrame(scores)

    plt.figure(figsize=(16, 12))
    sns.heatmap(scores_df, 
                annot=True, 
                cmap='viridis', 
                linewidths=.5, 
                linecolor='white', 
                annot_kws={'size': 8, 'weight': 'bold', 'color': 'black'},
                cbar_kws={'orientation': 'vertical', 'pad': 0.05, 'shrink': 0.75})
    
    #plt.title('Scores Heatmap', fontsize=18, weight='bold', pad=20)
    plt.xlabel('Columns', fontsize=12, weight='bold', labelpad=10)
    plt.ylabel('Index', fontsize=12, weight='bold', labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    sns.despine(left=True, bottom=True)
    plt.rcParams["figure.figsize"] = (12,6)
    plt.gcf().set_dpi(400)
    plt.savefig(title+".png", bbox_inches='tight')
    plt.savefig(title+".svg", bbox_inches='tight')
    plt.show()

def PlotMaps_Discret(quartis, scores, sp_file_name, main_state=None):
    
    # Convert quartis and scores into DataFrames
    quartis_df = pd.DataFrame(quartis)
    scores_df = pd.DataFrame(scores)
    
    # Set index name for both DataFrames
    quartis_df.index.name = 'SIGLA_UF'
    scores_df.index.name = 'SIGLA_UF'
    
    # Ensure that the index is of type string
    quartis_df.index = quartis_df.index.astype(str)
    scores_df.index = scores_df.index.astype(str)
    
    # Read the shapefile
    shapefile = gpd.read_file(sp_file_name)
    shapefile['SIGLA_UF'] = shapefile['SIGLA_UF'].astype(str)
    
    # Sort shapefile and DataFrames by 'SIGLA_UF'
    quartis_df.sort_index(inplace=True)
    scores_df.sort_index(inplace=True)
    shapefile.sort_values(by='SIGLA_UF', inplace=True)
    
    # Merge the shapefile with quartis and scores
    merged_shapefile = shapefile.merge(quartis_df, how='left', left_on='SIGLA_UF', right_index=True)
    merged_shapefile = merged_shapefile.merge(scores_df, how='left', left_on='SIGLA_UF', right_index=True)
    
    # Check for any NaN values after the merge
    if merged_shapefile.isnull().any().any():
        print("Warning: There are NaN values in the merged data. Please check the matching between states and the data.")
    
    # Reproject the geometries to a projected CRS (e.g., UTM Zone 23S for Brazil)
    merged_shapefile = merged_shapefile.to_crs(epsg=31983)
    
    # Plot each quartile and score pair
    for Q_score_name, Z_score_name in zip(quartis_df.columns, scores_df.columns):
                
        fig, ax = plt.subplots(figsize=(12, 12))
        
        merged_shapefile.plot(ax=ax, column=Q_score_name, legend=True, categorical=True, cmap='plasma', edgecolor='white', linewidth=1)
        ax.set_axis_off()
        
        # Plot Main state highlight
        if main_state is not None:
            merged_shapefile[merged_shapefile['SIGLA_UF'] == main_state].plot(ax=ax, edgecolor='black', linewidth=6, facecolor='none')
        
        # Add label with z-score
        merged_shapefile["center"] = merged_shapefile["geometry"].centroid
        za_points = merged_shapefile.copy()
        za_points.set_geometry("center", inplace=True)
        texts = []
        for x, y, label in zip(za_points.geometry.x, za_points.geometry.y, merged_shapefile[Z_score_name]):
            texts.append(ax.annotate(round(label, 3), xy=(x, y), color='black', fontsize=16,
                                     path_effects=[pe.withStroke(linewidth=4, foreground="white")],
                                     ha='center'))
        
        adjust_text(texts, force_points=0.3, force_text=0.8, expand=(2.5, 2.5), 
                    arrowprops=dict(arrowstyle="->", color='black', lw=0.5))
        
        fig.set_dpi(400)
        fig.savefig('Mapa_' + Q_score_name + '.png', bbox_inches='tight')
        
        plt.show()
    
    
        
def PlotMaps(scores, sp_file_name, main_state=None):
    scores_df = pd.DataFrame(scores)
    scores_df.index.name = 'SIGLA_UF'
    scores_df.index = scores_df.index.astype(str)
    
    shapefile = gpd.read_file(sp_file_name)
    shapefile['SIGLA_UF'] = shapefile['SIGLA_UF'].astype(str)
    
    # Merge the shapefile with the scores DataFrame
    merged_shapefile = shapefile.merge(scores_df, how='left', left_on='SIGLA_UF', right_index=True)
    
    # Reproject the geometries to a projected CRS (e.g., UTM Zone 23S for Brazil)
    merged_shapefile = merged_shapefile.to_crs(epsg=31983)
    
    for score_name in scores_df.columns:
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plotting the scalar values using the 'viridis' colormap
        merged_shapefile.plot(ax=ax, column=score_name, legend=True, cmap='viridis', edgecolor='white', linewidth=1)
        ax.set_axis_off()
        
        # Plot Main state highlight if specified
        if main_state is not None:
            merged_shapefile[merged_shapefile['SIGLA_UF'] == main_state].plot(ax=ax, edgecolor='black', linewidth=6, facecolor='none')
        
        ax.set_title(score_name, fontsize=15)
        fig.set_dpi(400)
        fig.savefig('MapaZ_' + score_name + '.png', bbox_inches='tight')

def SpatilAnalysis(df, main_state, plotNeighborhood=False,  plotMoran=False, plotMaps=False):
    main_values = Dictionaries.main_values
    neighbors = Dictionaries.neighbors
    fleet = Dictionaries.fleet
    
    scores_perAtt = {}
    localMoranIndex_perAtt = {}
    quartil_score = {}
    globalMoranIndex_perAtt = {}    
    
    attribute_isCorrelated = {}
    for attribute,values in main_values.items():
        
        for value in values:
            
            # Keep only the main classes of certain attribute
            filtered_df  = keep_rows(df,attribute,[value])
            
            # Create a time series for every state
            valueTimeSeries_Dataset = tst.CreateTimeSeriesDataset(filtered_df,
                                                                  attribute='uf',
                                                                  values=list(neighbors.keys()), 
                                                                  granularity=time_window_size,
                                                                  count=time_window_type)
            
            if valueTimeSeries_Dataset is not None:
                # Divide the time series by each fleet size 
                valueTimeSeries_Dataset = tsac.normalize_by_dict(valueTimeSeries_Dataset,fleet)
                                
                # Calculate the Moran Indexes
                globalMoranIndex, localMoranIndex_uf, Z_uf = tsac.MoranIndex(valueTimeSeries_Dataset,
                                                                            neighbors,
                                                                            alpha=2)
                
                # Make plots
                if plotNeighborhood:
                    # Plot the time series vs the national and regional mean
                    plt.PlotNeighboorhod(valueTimeSeries_Dataset,
                                          neighbors,
                                          main_state,
                                          title="neighborhood_timeSeries_byCategory_"+attribute+"_"+value)
        
                
                #Plot the Moran Diagram: state Z-score vs neighbors Z-score
                quartil_uf = plt.MoranPlot(Z_uf,
                                            neighbors,
                                            globalMoranIndex,
                                            title="Moran_Diag"+attribute+"_"+value, main_state=main_state)
                
                attribute_isCorrelated[value] = False
                
                if Validate(globalMoranIndex, localMoranIndex_uf, gamma=0.1):
                    attribute_isCorrelated[value] = True
                    
                    # Save findings for plot
                    scores_perAtt[attribute+"="+value] = Z_uf
                    quartil_score["Quartis_"+attribute+"="+value] = quartil_uf
                    localMoranIndex_perAtt[attribute+"="+value] = localMoranIndex_uf
                    globalMoranIndex_perAtt[attribute+"="+value] = globalMoranIndex
                    
            
                
    if plotMaps:
        PlotMaps_Discret(quartil_score,scores_perAtt, sp_file_name)
        PlotMaps(scores_perAtt,sp_file_name)
        
    ret1 =  pd.DataFrame(scores_perAtt)
    ret2 =  pd.DataFrame(localMoranIndex_perAtt)
    ret3 =  pd.DataFrame.from_dict(globalMoranIndex_perAtt, orient='index', columns=['Global'])
        
    return ret1,ret2,ret3

def Plot_Comparison(att, scores_uf_PP, scores_uf_Pa):
    ax = plt.gca()

    # Match the columns order
    x_values = scores_uf_PP.values
    y_values = scores_uf_Pa.values
    
    x_min = min(x_values)
    x_max = max(x_values)
    
    # Plot the Regressed Line
    slope, intercept, r, p, std_err = stats.linregress(x_values, y_values)
    x_line = np.linspace(x_min, x_max, 100)
    ax.plot(x_line, slope*x_line+ intercept , color='#3b3b3b', linestyle='--', linewidth=2)

    # Plot perfect line
    ax.plot(x_line, x_line, color='#3b3b3b', linestyle=':', linewidth=2)

    # Marke quarters
    ax.axvline(x=0, color='grey', linestyle='--')
    ax.axhline(y=0, color='grey', linestyle='--')
    
    # Plot the points with increased dot size
    ax.plot(x_values, y_values, 'o', alpha=0.8, markeredgecolor='k', markersize=12, color='#0d0887')
    
    
    if main_state != "": 
        main_state_x = scores_uf_PP[main_state]
        main_state_y = scores_uf_Pa[main_state]
        ax.plot([main_state_x], [main_state_y], 'o', alpha=0.8, markeredgecolor='k', markersize=18, color='#b73779')

    # Add text labels to points
    texts = [ax.text(x_value,y_value, uf, fontsize=12) for uf, x_value,  y_value in zip(scores_uf_PP.index, x_values , y_values)]

    ax.set_xlabel("Pre-Pandemic scores", fontsize=12)
    ax.set_ylabel("Pandemic scores", fontsize=12)

    # Increase tick size
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Adjust text to avoid overlap
    adjust_text(texts, expand=(3, 3), 
                arrowprops=dict(arrowstyle='->', color='red'))

    plt.rcParams["figure.figsize"] = (12, 6)
    plt.gcf().set_dpi(600)
    plt.savefig("Z_change_"+att + ".png")
    plt.savefig("Z_change_"+att + ".svg")
    plt.show()
    
    
    
    
    
    
    
    
#######################
# ANALYST USER INPUTS
#######################

# Load Accidents Instances - Pre Processed, so that classes are previously clustered
csv_file_name = r'PreProc_Data.csv'
# Load the State's cars fleet
sp_file_name = r'BR_UF_2022.shp'

# Time window setup, time series will be formed counting all accidents in every time-window
time_window_type = 'months'
time_window_size = 1

main_state = 'SP'
alpha = 2
#-----------------------






#######################
# COMPUTATIONS
#######################


# Load data
df = pd.read_csv(csv_file_name,low_memory=False, parse_dates=['data_inversa'])

# Create time series dataframe for each period
df_CE_PrePandemics = tst.filter_dates_between(df, '2018-03-15', '2020-03-15', date_column='data_inversa')
df_CE_Pandemics = tst.filter_dates_between(df, '2020-03-16', '2022-04-22', date_column='data_inversa')
df_CE_AfterPandemics = tst.filter_dates_between(df, '2022-04-23', '2024-04-23', date_column='data_inversa')

# Compute scores for each period
scores_perAtt_PP, localMoranIndex_perAtt_PP, globalMoranIndex_perAtt_PP = SpatilAnalysis(df_CE_PrePandemics, main_state, plotNeighborhood=True)
scores_perAtt_Pa, localMoranIndex_perAtt_P, globalMoranIndex_perAtt_P = SpatilAnalysis(df_CE_Pandemics, main_state, plotNeighborhood=True)
scores_perAtt_AP, localMoranIndex_perAtt_AP, globalMoranIndex_perAtt_AP = SpatilAnalysis(df_CE_AfterPandemics, main_state)

# Match columns, avoiding trying to compare attributes that only occured in one of the periods
scores_perAtt_PP = scores_perAtt_PP[scores_perAtt_Pa.columns]

"""
# For each attribute, plot the 
for (att, scores_uf_PP), (_,scores_uf_Pa) in zip(scores_perAtt_PP.iteritems(), scores_perAtt_Pa.iteritems()):
    Plot_Comparison(att, scores_uf_PP, scores_uf_Pa)
"""
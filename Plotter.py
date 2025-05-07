import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patheffects as pe
import matplotlib.dates as mdates
import seaborn as sns
from adjustText import adjust_text
import geopandas as gpd
import TimeSeriesTransformation as tst
import numpy as np
import TimeSeriesAutoCorrelation as tsac

import os

import re

def sanitize_filename(name):
    # Substitui apenas os caracteres inválidos no nome do arquivo
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def save_figure(title):
    directory = os.path.dirname(title)
    filename = os.path.basename(title)

    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    safe_filename = sanitize_filename(filename)
    full_path = os.path.join(directory, safe_filename)

    plt.savefig(full_path + ".svg", bbox_inches='tight')
    plt.savefig(full_path + ".png", bbox_inches='tight')
    plt.close()
    
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
    save_figure(title)

def PlotMaps_Discret(quartis, scores, sp_file_name, main_state=None, title="QuartilMap"):
    
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
        
        save_figure(title + Q_score_name)
        
    
    
        
def PlotMaps(scores, sp_file_name, main_state=None, title="Map"):
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
        
        save_figure(title + score_name)

"""def PlotTimeSeries(totalTimeSeries, count='days', granularity=7, title="Total", vline_date=None, vline_label=None):
    totalTimeSeries = tst.sort_time_series(totalTimeSeries)
    totalTimeSeries = tst.granular_resample(totalTimeSeries, count, granularity)
    if not pd.api.types.is_datetime64_any_dtype(totalTimeSeries.index):
        totalTimeSeries.index = pd.to_datetime(totalTimeSeries.index, format='%d %b %Y')
        
    # Calculate the moving average for one year (12 months)
    window = round(365/granularity) if count == 'days' else round(12/granularity)    
    moving_average = totalTimeSeries.rolling(window=window, center=True).mean()
    
    # Set up the first figure for the total occurrences plot
    plt.figure(figsize=(8, 6), dpi=300)
    palette = sns.color_palette("magma", 3)
    
    # Plot the total occurrences
    sns.lineplot(x=totalTimeSeries.index, y=totalTimeSeries, color=palette[0], label="Total in Brazil every "+str(granularity)+" "+count, linewidth=3)
    # Plot the moving average with a dashed line
    sns.lineplot(x=moving_average.index, y=moving_average, color='indianred', label='Yearly Moving Average', linestyle='--', linewidth=5)
 
    # Format x-axis ticks
    date_format = mdates.DateFormatter('%d %b %Y')
    plt.gca().xaxis.set_major_formatter(date_format)
    
    plt.xlabel('Date')
    plt.ylabel('Total occurrences every '+str(granularity)+" "+count)
    plt.ylim(min(totalTimeSeries) * 0.8, max(totalTimeSeries) * 1.1)
    plt.xticks(totalTimeSeries.index[::14], rotation=45)
    plt.grid(True)
    plt.legend()
    
    if vline_date:
        vline_date = pd.to_datetime(vline_date, format="%d %b %Y")

        # TIMESERIES VERTICAL LINE
        plt.axvline(vline_date, color='black', linestyle=':', linewidth=3)
        if vline_label:
            x_shift = pd.DateOffset(days=15)
            plt.text(vline_date + x_shift, plt.gca().get_ylim()[0] + 15, vline_label, color='black', fontsize=14)
        
    plt.tight_layout()
    
    save_figure(title + "_timeseries")"""


def PlotTimeSeries(totalTimeSeries, count='days', granularity=7, title="Total", vline_dates=None, vline_labels=None):
    totalTimeSeries = tst.sort_time_series(totalTimeSeries)
    totalTimeSeries = tst.granular_resample(totalTimeSeries, count, granularity)

    if not pd.api.types.is_datetime64_any_dtype(totalTimeSeries.index):
        totalTimeSeries.index = pd.to_datetime(totalTimeSeries.index, format='%d %b %Y')

    # Calculate the moving average for one year (12 months)
    window = round(365 / granularity) if count == 'days' else round(12 / granularity)
    moving_average = totalTimeSeries.rolling(window=window, center=True).mean()

    # Set up the first figure for the total occurrences plot
    plt.figure(figsize=(8, 6), dpi=300)
    palette = sns.color_palette("magma", 3)

    # Plot the total occurrences
    sns.lineplot(x=totalTimeSeries.index, y=totalTimeSeries, color=palette[0],
                 label="Total in Brazil every " + str(granularity) + " " + count, linewidth=3)

    # Plot the moving average
    sns.lineplot(x=moving_average.index, y=moving_average, color='indianred', label='Yearly Moving Average',
                 linestyle='--', linewidth=5)

    # Format x-axis ticks
    date_format = mdates.DateFormatter('%d %b %Y')
    plt.gca().xaxis.set_major_formatter(date_format)

    plt.xlabel('Date')
    plt.ylabel('Total occurrences every ' + str(granularity) + " " + count)
    plt.ylim(min(totalTimeSeries) * 0.8, max(totalTimeSeries) * 1.1)
    plt.xticks(totalTimeSeries.index[::14], rotation=45)
    plt.grid(True)
    plt.legend()

    # Plot multiple vertical lines if provided
    if vline_dates:
        # Convert single string to list
        if isinstance(vline_dates, str):
            vline_dates = [vline_dates]
        vline_dates = pd.to_datetime(vline_dates, format="%d %b %Y")

        if vline_labels is None:
            vline_labels = [''] * len(vline_dates)
        elif isinstance(vline_labels, str):
            vline_labels = [vline_labels]

        for date, label in zip(vline_dates, vline_labels):
            plt.axvline(date, color='black', linestyle=':', linewidth=3)
            if label:
                x_shift = pd.DateOffset(days=15)
                plt.text(date + x_shift, plt.gca().get_ylim()[0] + 15, label, color='black', fontsize=14)

    plt.tight_layout()
    save_figure(title + "_timeseries")


# 'att_scores' must be a two column dataframe with N rows (samples)
def plotRegression(att_scores, slope, intercept, main_states=[], title="Regression"):
    
    ax = plt.gca()
    
    x_values = att_scores.iloc[:,0]
    y_values = att_scores.iloc[:,1]
    
    # Plot the Regressed Line vs Perfect Line
    x_min = x_values.min()
    x_max = x_values.max()
    y_min = y_values.min()
    y_max = y_values.max()
    max_ = round(max(np.abs(x_min), np.abs(x_max), np.abs(y_min), np.abs(y_max)) * 1.2, 1)
    x_line = np.linspace(-max_, max_, 100)
    ax.plot(x_line, slope*x_line+ intercept , color='royalblue', linestyle='--', linewidth=2, label="Regressed case")
    ax.plot(x_line, x_line, color='black', linestyle=':', linewidth=2, label="Constant in time case")
    ax.legend()
    # Marke quarters
    ax.axvline(x=0, color='silver', linestyle='--')
    ax.axhline(y=0, color='silver', linestyle='--')

    ax.set_xlim(-max_ , max_)
    ax.set_ylim(-max_, max_)
    # Plot the samples
    ax.plot(x_values.values, y_values.values, 'o', alpha=0.8, markeredgecolor='k', markersize=12, color='#0d0887')
    
    for main_state in main_states: 
        main_state_x = x_values[main_state]
        main_state_y = y_values[main_state]
        ax.plot([main_state_x], [main_state_y], 'o', alpha=0.8, markeredgecolor='k', markersize=18, color='#b73779')

    # Add text labels to points
    texts = [ax.text(x_value,y_value, uf, fontsize=12) for uf, x_value,  y_value in zip(att_scores.index, x_values.values , y_values.values)]

    ax.set_xlabel("Pre-Lockdown scores", fontsize=12)
    ax.set_ylabel("Lockdown scores", fontsize=12)

    # Increase tick size
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Adjust text to avoid overlap
    adjust_text(texts, expand=(3, 3), arrowprops=dict(arrowstyle='->', color='red'))

    plt.rcParams["figure.figsize"] = (12, 6)
    plt.gcf().set_dpi(600)
    
    save_figure(title)
    
def PlotNeighboorhod(valueTimeSeries_Dataset,neighbors,main_state, title):
    uf_list = list(neighbors.keys())
        
    # Plot the main state line with highlight
    state_valueTimeSeries = valueTimeSeries_Dataset[main_state]
    sns.lineplot(data=state_valueTimeSeries,label=main_state, linewidth=4.5,  color='#b73779')
    

    
    mean_series = valueTimeSeries_Dataset.mean(axis=1)
    std_series = valueTimeSeries_Dataset.std(axis=1)


    uf_list = neighbors[main_state]
    neighbors_valueTimeSeries_Dataset = valueTimeSeries_Dataset[uf_list]
    mean_neighbors = neighbors_valueTimeSeries_Dataset.mean(axis=1) 
    #std_neighbors = neighbors_valueTimeSeries_Dataset.std(axis=1)
    
    sns.lineplot(x=mean_series.index, y=mean_series, color='black', label='National Mean', linewidth=3)
    sns.lineplot(x=mean_neighbors.index, y=mean_neighbors, color='#21918c', label='Neighborhood Mean', linewidth=3, linestyle='dashed')
    
    plt.fill_between(mean_series.index, mean_series - std_series, mean_series + std_series, color='grey', alpha=0.5, label='National std.')
    #plt.fill_between(mean_neighbors.index, mean_neighbors - std_neighbors, mean_neighbors + std_neighbors, color='red', alpha=0.4)

    
    # Customize labels and title
    plt.xlabel('Date')
    plt.ylabel('Occurences')
    #plt.title(f'Time Series of neighbor states {attribute}')

    # Set x-axis ticks
    plt.xticks(ticks=valueTimeSeries_Dataset.index[::7], rotation=45)  # Adjust the frequency as needed (e.g., every 30th date)

    # Show plot
    plt.tight_layout()
    plt.grid(True)
    plt.rcParams["figure.figsize"] = (12,6)
    plt.gcf().set_dpi(400)
    save_figure(title)
    
    

def MoranPlot(Z_uf_dict, neighbors, MoranIndex, title="Spatial Correlation Quarters", main_state="", dot_size=12, text_size=12):
    ax = plt.gca()

    # Set light grey background
    ax.set_facecolor('lightgrey')

    # Normalize the data
    z_values = list(Z_uf_dict.values())
    neighborhood_avg = tsac.getNeighborhoodAvg(Z_uf_dict, neighbors)
    y_values = list(neighborhood_avg.values())

    z_values_norm = (np.array(z_values) - np.mean(z_values)) / np.std(z_values)

    # Plot the Moran's Index line
    x_min = min(z_values_norm)
    x_max = max(z_values_norm)
    x_line = np.linspace(x_min, x_max, 100)
    y_line = MoranIndex * x_line
    ax.plot(x_line, y_line, color='#3b3b3b', linestyle='--', linewidth=2)
    # Plot a vertical line for Moran's Index
    ax.axvline(x=0, color='grey', linestyle='--')
    ax.axhline(y=0, color='grey', linestyle='--')
    # Plot the points with increased dot size
    ax.plot(z_values_norm, y_values, 'o', alpha=0.8, markeredgecolor='k', markersize=dot_size, color='#0d0887')
    
    
    if main_state != "": 
        uf_x = (Z_uf_dict[main_state] - np.mean(z_values)) / np.std(z_values)
        uf_y = neighborhood_avg[main_state]
        ax.plot([uf_x], [uf_y], 'o', alpha=0.8, markeredgecolor='k', markersize=dot_size + 6, color='#b73779')

    # Add text labels to points
    texts = [ax.text(z_values_norm[i], y_values[i], uf, fontsize=text_size) for i, uf in enumerate(Z_uf_dict)]


    ax.set_xlabel("z_score = (score - mean) / std", fontsize=text_size)
    ax.set_ylabel("neighborhood z_score", fontsize=text_size)
    ax.set_title(title, fontsize=text_size + 2)


    # Increase tick size
    ax.tick_params(axis='both', which='major', labelsize=text_size)

    # Adjust text to avoid overlap
    adjust_text(texts, expand=(3, 3), 
                arrowprops=dict(arrowstyle='->', color='red'))

    plt.rcParams["figure.figsize"] = (12, 6)
    plt.gcf().set_dpi(600)
    
    save_figure(title)
    
def PlotComparison(score_regressions, scores_findings, common_scores_PreLockdown, common_scores_Lockdown, title="", annomaliesOnly=False):
    if annomaliesOnly:
        scores_findings = scores_findings[scores_findings['Anomaly'] != '']

    # Plot the regression of the attribute's score, highlighting the states with anomalies
    for index, row in score_regressions.iterrows():
        
        attribute = row['Attribute']
        class_ = row['Class']
        
        att_scores_Period1 = common_scores_PreLockdown[
            (common_scores_PreLockdown['Attribute'] == attribute) & (common_scores_PreLockdown['Class'] == class_)
            ].drop(columns=['Attribute', 'Class']).iloc[0]
        
        att_scores_Period2 = common_scores_Lockdown[
            (common_scores_Lockdown['Attribute'] == attribute) & (common_scores_Lockdown['Class'] == class_)
            ].drop(columns=['Attribute', 'Class']).iloc[0]
        
        # Make sure they have same elements
        att_scores = pd.concat([att_scores_Period1, att_scores_Period2], axis=1)
        
        
        
        att_slope =  score_regressions[
            (score_regressions['Attribute'] == attribute) & (score_regressions['Class'] == class_)
            ]['Slope'].iloc[0]
        
        att_interception = 0
        
        main_states = scores_findings[
            (scores_findings['Attribute'] == attribute) & (scores_findings['Class'] == class_)
            ]['Anomaly'].to_list()
        
        plotRegression( 
            att_scores,
            att_slope,
            att_interception,
            main_states=main_states,
            title=title+"Regression_"+attribute+class_)
        
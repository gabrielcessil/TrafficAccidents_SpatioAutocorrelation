import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from scipy import stats
import pandas as pd
import pylab as py
import statsmodels.api as sm
from scipy.stats import shapiro

import TimeSeriesAutoCorrelation as tsac
import Dictionaries
import TimeSeriesTransformation as tst
import Plotter


def keep_rows(df, attribute, values): return df[df[attribute].isin(values)]

def Validate_SpatialAutocorrelation_fromExpected(globalIndex, localIndexes, gamma=0.05):
    E = - 1/ (len(localIndexes)-1)
    teste = globalIndex < E*(1+gamma) or E*(1-gamma) < globalIndex
    return teste
    

def Validate_SpatialAutocorrelation_fromProbability(valueTimeSeries_Dataset, n=1000, alpha=0.01):
    
    def permute_column_names(df):
        permuted_df = df.copy()
        permuted_columns = np.random.permutation(df.columns)
        permuted_df.columns = permuted_columns
        return permuted_df
    
    random_Ig = []
    random_Il = []
    for i in range(n):
        random_valueTimeSeries_Dataset = permute_column_names(valueTimeSeries_Dataset)
        globalMoranIndex, localMoranIndex_uf, Z_uf = tsac.MoranIndex(random_valueTimeSeries_Dataset,
                                                                     neighbors=Dictionaries.neighbors,
                                                                     alpha=2)
        random_Ig.append(globalMoranIndex)
        random_Il.append(localMoranIndex_uf)
        
    t_stat, p_value = stats.ttest_1samp(random_Ig, 0)
    return p_value < alpha

def SpatilAnalysis(df, plotNeighborhood=False,  plotMoran=False, plotMaps=False, granularity=7, count='days', shp_file_name="",  main_state="", title=""):
    # Classes and Attributes
    main_values = Dictionaries.main_values
    
    # Neighborhood structure
    neighbors = Dictionaries.neighbors
    
    # Local's car fleet
    fleet = Dictionaries.fleet
    
    scores_perAtt = []
    localMoranIndex_perAtt = []
    quartil_score = []
    globalMoranIndex_perAtt = []
    
    attribute_isCorrelated = {}
    
    # For each attribute
    for attribute,values in main_values.items():
        
        # For each class of certain attribute
        for value in values:
            
            # Keep only the main pre-defined classes of certain attribute
            filtered_df  = keep_rows(df,attribute,[value])
            
            # Create a time series for every state
            valueTimeSeries_Dataset = tst.CreateTimeSeriesDataset(filtered_df,
                                                                  attribute='uf',
                                                                  values=list(neighbors.keys()), 
                                                                  granularity=granularity,
                                                                  count=count)
            
            # If the creating of time series dataset was possible
            if valueTimeSeries_Dataset is not None:
                                
                # Divide the time series by each fleet size 
                valueTimeSeries_Dataset = tsac.normalize_by_dict(valueTimeSeries_Dataset,fleet)
                
                # Calculate the Global Moran, Local Moran Indexes and Dissimilarity score for each state 
                globalMoranIndex, localMoranIndex_uf, Z_uf = tsac.MoranIndex(
                    valueTimeSeries_Dataset,
                    neighbors,
                    alpha=2)
                
                # Classify the results
                quartil_uf = tsac.get_quartis(Z_uf, neighbors)
                
                attribute_isCorrelated[value] = False
                if Validate_SpatialAutocorrelation_fromExpected(globalMoranIndex, localMoranIndex_uf, gamma=0.1):
                    attribute_isCorrelated[value] = True
                    
                    scores_perAtt.append( {"Attribute":attribute,"Class": value, **Z_uf} )
                    quartil_score.append( {"Attribute":attribute,"Class": value, **quartil_uf} )
                    localMoranIndex_perAtt.append( {"Attribute":attribute,"Class": value, **localMoranIndex_uf} ) 
                    globalMoranIndex_perAtt.append( {"Attribute":attribute,"Class": value,"Value":globalMoranIndex}  )
                
                
                
                # MAKE PLOTS:
        
                if plotNeighborhood:
                    if main_state != "": states = [main_state]
                    else: states=valueTimeSeries_Dataset.columns.tolist()
                    
                    # Plot the time series vs the national and regional mean for each state
                    for state in states:    
                        Plotter.PlotNeighboorhod(
                            valueTimeSeries_Dataset,
                            neighbors,
                            main_state=state,
                            title=title+"/Neighborhood_TimeSeries/"+attribute+"_"+value+"/"+state)
                    
                if plotMoran:
                    Plotter.MoranPlot( 
                        Z_uf,
                        neighbors,
                        globalMoranIndex,
                        title=title+"/Moran_Diag/"+attribute+"_"+value, 
                        main_state=main_state)
                
    if plotMaps:
        Plotter.PlotMaps_Discret(quartil_score,scores_perAtt, shp_file_name, title=title+"/Maps_QuartilScore/")
        Plotter.PlotMaps(scores_perAtt,shp_file_name, title=title+"/Map_DissScore/")
        
    
    # Create scores dataframes 
    ret1 =  pd.DataFrame(scores_perAtt)
    ret2 =  pd.DataFrame(localMoranIndex_perAtt)
    ret3 =  pd.DataFrame(globalMoranIndex_perAtt)
        
    return ret1,ret2,ret3, attribute_isCorrelated

import os
def shapiro_wilk(y_values, predicted_values, filename=""):
    # Ensure directory exists
    if filename:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    shapiro_result = shapiro(predicted_values - y_values)
    statistic = shapiro_result.statistic
    p_value = shapiro_result.pvalue

    if filename!="":
        sm.qqplot(predicted_values - y_values, line='q')
        py.text(
            0.05, 0.95,
            f'Shapiro-Wilk Test:\nStatistic = {statistic:.4f}\np-value = {p_value:.4f}',
            transform=py.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
        )
        py.savefig(filename + ".png", dpi=300, bbox_inches='tight')
        py.show()
    return statistic, p_value

def getAnomalies(y_values, predicted_values):
    
    # Calculate residuals
    residuals = abs(y_values - predicted_values)
    
    mean_residual = residuals.mean()
    std_residual = residuals.std()
    # Calculate Z-scores of residuals
    z_scores = (residuals - mean_residual) / std_residual
    
    # Identify anomalies (Z-score > 3 or < -3)
    anomalies = residuals[np.abs(z_scores) > 3].index.tolist()

    return anomalies


def ProfileChangeAnalysis(scores_PreEvent, scores_AfterEvent, r2_threshold=0.6,p_threshold=0.05, filename=""):

    regressions_findings = []
    anomalous_findings = []

    for (_,PE_location_values), (_,AE_location_values) in zip(scores_PreEvent.iterrows(), scores_AfterEvent.iterrows()):


        att_name = PE_location_values['Attribute']
        class_name = PE_location_values['Class']


        att_score_AfterEvent = AE_location_values.drop(['Attribute', 'Class'])
        att_score_PreEvent = PE_location_values.drop(['Attribute', 'Class'])
        
        att_scores = pd.concat(
            [att_score_PreEvent, att_score_AfterEvent], 
            axis=1, 
            keys=['Scores Before Lockdown', 'Scores After Lockdown'])
        

        # Regression
        x_values = att_scores.iloc[:, 0]
        y_values = att_scores.iloc[:, 1]
        model = LinearRegression(fit_intercept=False)
        model.fit(x_values.values.reshape(-1, 1), y_values.values.reshape(-1, 1))
        slope = model.coef_[0][0]
        predicted_values = slope * x_values

        # Measure how well was the model fitted
        r2 = r2_score(y_values, predicted_values)
        mae = mean_absolute_error(y_values, predicted_values)

        # Residuals distribution validation
        statistic, p_value = shapiro_wilk(y_values, predicted_values)#, filename=filename +"Shapiro_"+ att_name + class_name)

        regressions_findings.append({
            'Attribute': att_name,
            'Class': class_name,
            "Slope": np.round(slope, 3),
            "Temporal Linearity (R2)": round(r2, 3),
            "Residuals Normality (Shapiro-Wilk p-value)": round(p_value,5),
            "Acceptable Temporal Linearity" : r2 > r2_threshold,
            # If p_value is too low (unlikely), the null hypothesis (normal distribution) can be denied:
            # so we say that the data is significantly far from a normal distribution
            "Acceptable Residuals Normality": p_value > p_threshold
        })

        # Validates the fitted model, ensuring that the anomaly is got from a linear global behavior

        # Get the locations with anomalies
        anomalies = getAnomalies(y_values, predicted_values)

        # If there exist anomalies
        if anomalies:
            residuals = y_values - predicted_values
            anomalies_residuals = [str(round(residuals.loc[anomaly], 3)) for anomaly in anomalies]
            anomalies_slopes = [ str(round(y_values[anomaly]/x_values[anomaly],3)) for anomaly in anomalies]


            for an, an_res, an_slope in zip(anomalies, anomalies_residuals, anomalies_slopes):
                # Saving findings
                anomalous_findings.append({
                    'Attribute': att_name,
                    'Class': class_name,
                    "Anomaly": an,
                    "Anomaly Residual": an_res,
                    "Anomaly Slope": an_slope
                    })

    return pd.DataFrame(regressions_findings), pd.DataFrame(anomalous_findings)
    

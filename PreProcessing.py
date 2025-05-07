import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import math 
def convert_to_uniform_date(date_str):
    formats_to_try = ['%d/%m/%Y', '%Y-%m-%d', '%d/%m/%y']  # Add more formats as needed
    
    for fmt in formats_to_try:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            return date_obj.strftime('%d/%m/%Y')  # Convert to desired uniform format
        except ValueError:
            continue
    
    return date_str  # Return original date string if no format matches
# Function to calculate days since defined date
def calculate_days(date, defined_date):
    try:
        defined_date = datetime.strptime(defined_date, '%d/%m/%Y')
        
        if isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()
        elif isinstance(date, str):
            date = datetime.strptime(convert_to_uniform_date(date), '%d/%m/%Y')
            
        delta = date - defined_date
        return delta.days
    except Exception as e:
        print(f"Error processing date '{date}': {e}")
        return None

# Function to calculate months since defined date
def calculate_months(date, defined_date):
    try:
        defined_date = datetime.strptime(defined_date, '%d/%m/%Y')
        if isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()
        elif isinstance(date, str):
            date = datetime.strptime(convert_to_uniform_date(date), '%d/%m/%Y')
        
        delta = relativedelta(date, defined_date)
        return delta.years * 12 + delta.months
    except Exception as e:
        print(f"Error processing date '{date}': {e}")
        return None

# Function to calculate weeks since defined date
def calculate_weeks(date, defined_date):
    try:
        defined_date = datetime.strptime(defined_date, '%d/%m/%Y')
        if isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()
        elif isinstance(date, str):
            date = datetime.strptime(convert_to_uniform_date(date), '%d/%m/%Y')
        
        delta = date - defined_date
        return delta.days // 7
    except Exception as e:
        print(f"Error processing date '{date}': {e}")
        return None
    
    

def keep_rows(df, attribute, values):
    return df[df[attribute].isin(values)]


def createDistribution(df, attribute, values, title):
    # Get the counts and normalize them to percentages
    value_counts = df[attribute].value_counts(normalize=True) * 100
    
    # Create a DataFrame from the value counts
    temp_df = value_counts.reset_index()
    temp_df.columns = [attribute, 'Percentage']
    
    # If values is not empty, summarize values not in the list as 'others'
    if values:
        # Filter out the rows that are in the 'values' list
        filtered_df = temp_df[temp_df[attribute].isin(values)]
        
        # Sum up the percentage of values not in the list
        others_percentage = temp_df[~temp_df[attribute].isin(values)]['Percentage'].sum()
        
        # Create a new row for 'others'
        others_df = pd.DataFrame({attribute: ['Others'], 'Percentage': [others_percentage]})
        
        # Concatenate the filtered DataFrame with the 'others' DataFrame
        temp_df = pd.concat([filtered_df, others_df], ignore_index=True)
    
    # Add the dataset title
    temp_df['Dataset'] = title
    
    # Filter out zero percentages
    temp_df = temp_df[temp_df['Percentage'] > 0]
        
    return temp_df

def plot_distributions(df_dict, attribute, values, bar_vertical=True, title="distribution"):
    
    """
    Plots the distribution of unique values in a specified column of multiple DataFrames using percentages.
    
    Parameters:
    df_dict (dict): Dictionary where the key is the subtitle and the value is the DataFrame.
    attribute (str): The column name for which to plot the distributions.
    values (list): The list of values to keep in the DataFrame.
    bar_vertical (bool): If True, plot vertical bars; if False, plot horizontal bars.
    """
    
    combined_df = pd.DataFrame()
    
    # Combine data from all DataFrames into a single DataFrame
    for subtitle, df in df_dict.items():
        temp_df = createDistribution(df, attribute, values,subtitle)
        combined_df = pd.concat([combined_df, temp_df])

    
    if bar_vertical:
        plt.figure(figsize=(12, 6))

        ax = sns.barplot(x=attribute, y='Percentage', hue='Dataset', data=combined_df, palette="rocket_r")
        plt.ylabel('Percentage (%)', fontsize=16)
        #plt.title(f'Distribution of {attribute.capitalize()}', fontsize=20)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        
        # Annotate the bars with the percentage values
        for p in ax.patches:
            height = p.get_height()
            if height > 0:  # Only annotate if the height is greater than 0
                x = p.get_x() + p.get_width() / 2
                offset = 6
                ax.annotate(f'{height:.2f}%', 
                            (x, height + offset), 
                            ha='left', va='bottom', fontsize=14, color='black',
                            rotation = 45, 
                            rotation_mode = 'anchor')
        # Adjust frame limits
        max_percentage = combined_df['Percentage'].max()
        plt.ylim(0, max_percentage * 1.5)
        
        # Add the total number of occurrences for each dataset
        total_occurrences = {subtitle: len(df) for subtitle, df in df_dict.items()}
        total_text = '\n'.join([f'{key}: {value}' for key, value in total_occurrences.items()])
        plt.text(0.95, 0.95, f'Total occurrences:\n{total_text}', 
                 ha='right', va='top', transform=plt.gca().transAxes, fontsize=12.5, 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Adjust the legend to avoid overlap with the total occurrences
        ax.legend(bbox_to_anchor=(0.95, 0.75), loc='upper right', fontsize=14, frameon=False)
        
    else:
        plt.figure(figsize=(12, 8))

        ax = sns.barplot(y=attribute, x='Percentage', hue='Dataset', data=combined_df, palette="rocket_r")
        plt.xlabel('Percentage (%)', fontsize=16)
        #plt.title(f'Distribution of {attribute.capitalize()}', fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        # Annotate the bars with the percentage values
        for p in ax.patches:
            width = p.get_width()
            if width > 0:  # Only annotate if the width is greater than 0
                y = p.get_y() + p.get_height() / 2
                offset = 2.5
                ax.annotate(f'{width:.2f}%', 
                            (width + offset, y), 
                            ha='left', va='center', fontsize=14, color='black')
        
        
        # Adjust frame limits
        max_percentage = combined_df['Percentage'].max()
        plt.xlim(0, max_percentage * 1.25)
        
        # Add the total number of occurrences for each dataset
        total_occurrences = {subtitle: len(df) for subtitle, df in df_dict.items()}
        total_text = '\n'.join([f'{key}: {value}' for key, value in total_occurrences.items()])
        plt.text(0.95, 0.4, f'Total occurrences:\n{total_text}', 
                 ha='right', va='top', transform=plt.gca().transAxes, fontsize=12.5, 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Adjust the legend to avoid overlap with the total occurrences
        ax.legend(bbox_to_anchor=(0.95, 0.2), loc='upper right', fontsize=14, frameon=False)
        
    plt.tight_layout()
    plt.gcf().set_dpi(300)
    plt.savefig(title+".png", bbox_inches='tight')
    plt.savefig(title+".svg", bbox_inches='tight')
    plt.show()

def data_cleaning(file_name):
    # Read uploaded dataset
    df = pd.read_csv(file_name,low_memory=False, parse_dates=['data_inversa'])
    #import io
    #from google.colab import files
    #df = pd.read_csv(io.BytesIO(uploaded[file_name]))
    # Create a dictionary mapping the original cause names to their respective cluster names
   
    # PRE-PROCESSING PIPELINE 
    # Reset the index to maintain sequential index numbers (needed due the concatenation)
    df = df.reset_index(drop=True)

    df['data_inversa'] = df['data_inversa'].apply(convert_to_uniform_date)
    
    df.fillna('Unknown')
    return df

def countTime_dataframe(df,reference_date=None):
    # Apply functions to the DataFrame
    if reference_date is None:
        reference_date = df['data_inversa'].min()
        
    df['days_since_reference'] = df['data_inversa'].apply(calculate_days, defined_date=reference_date)
    df['weeks_since_reference'] = df['data_inversa'].apply(calculate_weeks, defined_date=reference_date)
    df['months_since_reference'] = df['data_inversa'].apply(calculate_months, defined_date=reference_date)
    
    return df

def reduce_dataframe(df):
    columns_to_keep = [
        "id",
        "data_inversa",
        "uf",
        "Accident_Cause",
        "Accident_Type",
        "Accident_Classification",
        "Day_Phase",
        "Weather_Condition",
        ]
    return df[columns_to_keep]


def translate_dataframe(df):
    # Function to translate values based on a dictionary
    def translate_column(column, translations):
        translated = []
        for value in column:
            if pd.isna(value):
                translated.append(None)
            else:
                translated.append(translations.get(value, value))
        return translated
    
    causa_acidente_translations = {
        'Não guardar distância de segurança': 'Unsafe Driving Practices and Disobedience',
        'Ultrapassagem indevida': 'Unsafe Driving Practices and Disobedience',
        'Transitar no acostamento': 'Unsafe Driving Practices and Disobedience',
        'Transitar no Acostamento': 'Unsafe Driving Practices and Disobedience',
        'Transitar na contramão': 'Unsafe Driving Practices and Disobedience',
        'Manobra de mudança de faixa': 'Unsafe Driving Practices and Disobedience',
        'Condutor deixou de manter distância do veículo da frente': 'Unsafe Driving Practices and Disobedience',
        'Conversão proibida': 'Unsafe Driving Practices and Disobedience',
        'Desrespeitar a preferência no cruzamento': 'Unsafe Driving Practices and Disobedience',
        'Trafegar com motocicleta (ou similar) entre as faixas': 'Unsafe Driving Practices and Disobedience',
        'Estacionar ou parar em local proibido': 'Unsafe Driving Practices and Disobedience',
        'Retorno proibido': 'Unsafe Driving Practices and Disobedience',
        'Participar de racha': 'Unsafe Driving Practices and Disobedience',
        'Velocidade incompatível': 'Unsafe Driving Practices and Disobedience',
        'Ultrapassagem Indevida': 'Unsafe Driving Practices and Disobedience',
        'Frear bruscamente': 'Unsafe Driving Practices and Disobedience',
        'Velocidade Incompatível': 'Unsafe Driving Practices and Disobedience',
        'Redutor de velocidade em desacordo': 'Unsafe Driving Practices and Disobedience',
        'Desobediência à sinalização': 'Unsafe Driving Practices and Disobedience',
        'Desobediência às normas de trânsito pelo condutor': 'Unsafe Driving Practices and Disobedience',
        'Condutor desrespeitou a iluminação vermelha do semáforo': 'Unsafe Driving Practices and Disobedience',
        'Modificação proibida': 'Unsafe Driving Practices and Disobedience',


        'Falta de atenção': 'Inattention and Distracted Driving',
        'Falta de Atenção à Condução': 'Inattention and Distracted Driving',
        'Reação tardia ou ineficiente do condutor': 'Inattention and Distracted Driving',
        'Ausência de reação do condutor': 'Inattention and Distracted Driving',
        'Condutor usando celular': 'Inattention and Distracted Driving',
        'Acessar a via sem observar a presença dos outros veículos': 'Inattention and Distracted Driving',

        'Ingestão de Álcool': 'Substance Influence',
        'Ingestão de álcool pelo condutor': 'Substance Influence',
        'Ingestão de álcool e/ou substâncias psicoativas pelo pedestre': 'Substance Influence',
        'Ingestão de substâncias psicoativas pelo condutor': 'Substance Influence',
        'Ingestão de Substâncias Psicoativas': 'Substance Influence',
        'Pedestre - Ingestão de álcool/ substâncias psicoativas': 'Substance Influence',
        'Ingestão de álcool': 'Substance Influence',

        'Falta de Atenção do Pedestre': 'Health and Fatigue',
        'Mal Súbito': 'Health and Fatigue',
        'Mal súbito do condutor': 'Health and Fatigue',
        'Transtornos Mentais (exceto suicidio)': 'Health and Fatigue',
        'Suicídio (presumido)': 'Health and Fatigue',
        'Condutor Dormindo': 'Health and Fatigue',
        'Dormindo': 'Health and Fatigue',

        'Desobediência às normas de trânsito pelo pedestre': 'Pedestrian Behavior',
        'Entrada inopinada do pedestre': 'Pedestrian Behavior',
        'Pedestre cruzava a pista fora da faixa': 'Pedestrian Behavior',
        'Pedestre andava na pista': 'Pedestrian Behavior',
        'Área urbana sem a presença de local apropriado para a travessia de pedestres': 'Pedestrian Behavior',
        'Transitar na calçada': 'Pedestrian Behavior',
        'Ingestão de álcool ou de substâncias psicoativas pelo pedestre': 'Pedestrian Behavior',

        'Defeito na Via': 'Infrastructure Issues',
        'Afundamento ou ondulação no pavimento': 'Infrastructure Issues',
        'Pista em desnível': 'Infrastructure Issues',
        'Acostamento em desnível': 'Infrastructure Issues',
        'Pista esburacada': 'Infrastructure Issues',
        'Curva acentuada': 'Infrastructure Issues',
        'Acesso irregular': 'Infrastructure Issues',
        'Obstrução na via': 'Infrastructure Issues',
        'Desvio temporário': 'Infrastructure Issues',
        'Obras na pista': 'Infrastructure Issues',
        'Declive acentuado': 'Infrastructure Issues',
        'Falta de acostamento': 'Infrastructure Issues',
        'Faixas de trânsito com largura insuficiente': 'Infrastructure Issues',
        'Demais falhas na via': 'Infrastructure Issues',
        'Falta de elemento de contenção que evite a saída do leito carroçável': 'Infrastructure Issues',
        'Deficiência do Sistema de Iluminação/Sinalização':'Infrastructure Issues',
        'Defeito na via': 'Infrastructure Issues',
        'Sistema de drenagem ineficiente': 'Infrastructure Issues',

        'Pista Escorregadia': 'Surface and Environmental Hazards',
        'Objeto estático sobre o leito carroçável': 'Surface and Environmental Hazards',
        'Acumulo de água sobre o pavimento': 'Surface and Environmental Hazards',
        'Acumulo de areia ou detritos sobre o pavimento': 'Surface and Environmental Hazards',
        'Acumulo de óleo sobre o pavimento': 'Surface and Environmental Hazards',
        'Demais Fenômenos da natureza':'Surface and Environmental Hazards',
        'Animais na Pista':'Surface and Environmental Hazards',


        'Sinalização da via insuficiente ou inadequada': 'Environmental Factors, Signage and Visibility',
        'Ausência de sinalização': 'Environmental Factors, Signage and Visibility',
        'Restrição de visibilidade em curvas horizontais': 'Environmental Factors, Signage and Visibility',
        'Sinalização mal posicionada': 'Environmental Factors, Signage and Visibility',
        'Iluminação deficiente': 'Environmental Factors, Signage and Visibility',
        'Restrição de Visibilidade': 'Environmental Factors, Signage and Visibility',
        'Chuva': 'Environmental Factors, Signage and Visibility',
        'Neblina': 'Environmental Factors, Signage and Visibility',
        'Fumaça': 'Environmental Factors, Signage and Visibility',
        'Semáforo com defeito': 'Environmental Factors, Signage and Visibility',
        'Restrição de visibilidade em curvas': 'Environmental Factors, Signage and Visibility',
        'Restrição de visibilidade em curvas verticais': 'Environmental Factors, Signage and Visibility',
        'Sinalização encoberta': 'Environmental Factors, Signage and Visibility',
        'Fenômenos da Natureza': 'Environmental Factors, Signage and Visibility',
        
        'Defeito mecânico em veículo': 'Mechanical Failures, Load and Maintenance',
        'Defeito Mecânico no Veículo': 'Mechanical Failures, Load and Maintenance',
        'Problema na suspensão': 'Mechanical Failures, Load and Maintenance',
        'Problema com o freio': 'Mechanical Failures, Load and Maintenance',
        'Demais falhas mecânicas ou elétricas': 'Mechanical Failures, Load and Maintenance',
        'Carga excessiva e/ou mal acondicionada': 'Mechanical Failures, Load and Maintenance',
        'Avarias e/ou desgaste excessivo no pneu': 'Mechanical Failures, Load and Maintenance',
        'Faróis desregulados': 'Mechanical Failures, Load and Maintenance',
        'Deficiência ou não Acionamento do Sistema de Iluminação/Sinalização do Veículo': 'Mechanical Failures, Load and Maintenance',
        'Deixar de acionar o farol da motocicleta (ou similar)': 'Mechanical Failures, Load and Maintenance',

        'Agressão Externa': 'Safety and Violence issues',
        'Obstrução Via tentativa Assalto': 'Safety and Violence issues',
        
        'Outras': 'Miscellaneous',
        '(null)': 'Miscellaneous'
    }
    
    # Translations dictionary for each column
    tipo_acidente_translations = {
        'Colisão lateral': 'Side collision',
        'Colisão lateral sentido oposto': 'Side collision',
        'Colisão lateral mesmo sentido': 'Side collision',
        'Colisão frontal': 'Head-on collision',
        
        'Saída de leito carroçável': 'Run-off-road collision',
        'Saída de Pista': 'Run-off-road collision',
        
        'Colisão traseira': 'Rear-end collision',
        'Engavetamento': 'Rear-end collision',
        
        'Tombamento': 'Rollover',
        'Capotamento': 'Rollover',
        
        'Colisão transversal': 'Transversal collision',
        'Colisão Transversal': 'Transversal collision',
        
        'Derramamento de carga': 'Cargo spill',
        'Derramamento de Carga': 'Cargo spill',
        
        'Colisão com objeto': 'Collision with object',
        'Colisão com objeto em movimento': 'Collision with object',
        'Colisão com objeto móvel': 'Collision with object',
        'Colisão com objeto fixo': 'Collision with object',
        'Colisão com objeto estático': 'Collision with object',
        
        'Colisão com bicicleta': 'Collision with bicycle',
        
        'Atropelamento de Pedestre': 'Person run over',
        'Atropelamento de pessoa': 'Person run over',
        'Atropelamento de animal': 'Animal run over',
        'Atropelamento de Animal': 'Animal run over',

        'Queda de motocicleta / bicicleta / veículo': 'Occupant falls from vehicle',
        'Queda de ocupante de veículo': 'Occupant falls from vehicle',
        
        'Incêndio': 'Fire',
        'Eventos atípicos': 'Atypical events',
        'Sinistro pessoal de trânsito': 'Personal traffic accident',
        
        'Danos eventuais': 'Eventual damage',
        'Danos Eventuais': 'Eventual damage'
        

    }
    
    
    classificacao_acidente_translations = {
        'Com Vítimas Feridas': 'With injured victims',
        'Com Vítimas Fatais': 'With fatal victims',
        'Sem Vítimas': 'Without victims',
        'Ignorado': 'Ignored',
        '(null)': 'Unknown'
    }
    
    fase_dia_translations = {
        'Plena Noite': 'Full night',
        'Plena noite': 'Full night',
        'Amanhecer': 'Dawn',
        'Pleno dia': 'Full day',
        'Anoitecer': 'Dusk',
        '(null)': 'Unknown'
    }
    
    condicao_metereologica_translations = {
        'Céu Claro': 'Clear sky',
        'Chuva': 'Rain',
        'Sol': 'Sun',
        'Nublado': 'Cloudy',
        'Garoa/Chuvisco': 'Drizzle',
        'Ignorado': 'Unknown',
        'Vento': 'Wind',
        'Nevoeiro/Neblina': 'Fog/Mist',
        'Granizo': 'Hail',
        'Neve': 'Snow',
        'Ceu Claro': 'Clear sky',
        'Nevoeiro/neblina': 'Fog/Mist',
        'Ignorada': 'Unknown',
        '(null)': 'Unknown'
    }
    
    dia_semana_translations = {
        'segunda-feira': 'Monday',
        'terça-feira': 'Tuesday',
        'quarta-feira': 'Wednesday',
        'quinta-feira': 'Thursday',
        'sexta-feira': 'Friday',
        'sábado': 'Saturday',
        'domingo': 'Sunday',
        'Sexta': 'Friday',
        'Sábado': 'Saturday',
        'Domingo': 'Sunday',
        'Segunda': 'Monday',
        'Terça': 'Tuesday',
        'Quarta': 'Wednesday',
        'Quinta': 'Thursday'
    }
    
    # Apply translations to each column
    df['tipo_acidente'] = translate_column(df['tipo_acidente'], tipo_acidente_translations)
    df['causa_acidente'] = translate_column(df['causa_acidente'], causa_acidente_translations)
    df['classificacao_acidente'] = translate_column(df['classificacao_acidente'], classificacao_acidente_translations)
    df['fase_dia'] = translate_column(df['fase_dia'], fase_dia_translations)
    df['condicao_metereologica'] = translate_column(df['condicao_metereologica'], condicao_metereologica_translations)
    df['dia_semana'] = translate_column(df['dia_semana'], dia_semana_translations)
    # Dictionary to translate column names
    column_name_translations = {
        'tipo_acidente': 'Accident_Type',
        'causa_acidente': 'Accident_Cause',
        'classificacao_acidente': 'Accident_Classification',
        'fase_dia': 'Day_Phase',
        'condicao_metereologica': 'Weather_Condition',
        'dia_semana': 'Day_of_Week'
    }
    # Apply translations to each column
    df.rename(columns=column_name_translations, inplace=True)
    return df







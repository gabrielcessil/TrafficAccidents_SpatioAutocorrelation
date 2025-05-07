main_values = {
    'Accident_Type':['Collision with object',
                     'Rear-end collision',
                    'Side collision',
                    'Transversal collision', 
                    'Rollover',
                    'Head-on collision',
                    'Run-off-road collision',
                    'Person run over',
                    'Animal run over',
                    'Occupant falls from vehicle',
                    ],
    
    'Accident_Classification':['Without victims',
                               'With injured victims',
                               'With fatal victims'],
    
    'Accident_Cause': [ 'Inattention and Distracted Driving',
                        'Unsafe Driving Practices and Disobedience',
                        'Miscellaneous',
                        'Mechanical Failures, Load and Maintenance',
                        'Substance Influence',
                        'Surface and Environmental Hazards',
                        'Health and Fatigue'],
    
    'Day_Phase':['Full day',
                 'Full night',
                 'Dawn',
                 'Dusk'],
    
    'Weather_Condition':['Clear sky',
                         'Cloudy',
                         'Rain',
                         'Sun'],
    
}



fleet = {
    "AC": 350273,
    "AP": 242574,
    "AM": 1130055,
    "PA": 2627090,
    "RO": 1197221,
    "RR": 275703,
    "TO": 874905,
    "AL": 1095144,
    "BA": 5120353,
    "CE": 3753826,
    "MA": 2132527,
    "PB": 1593744,
    "PE": 3568386,
    "PI": 1449658,
    "RN": 1554064,
    "SE": 951523,
    "ES": 2357061,
    "MG": 13481706,
    "RJ": 7705012,
    "SP": 33264096,
    "PR": 8838800,
    "RS": 8075318,
    "SC": 6189405,
    "DF": 2083081,
    "GO": 4726950,
    "MT": 2695548,
    "MS": 1893634
}

neighbors = {
    "AC": ["AM", "RO"], 
    "AL": ["PE", "SE", "BA"], 
    "AP": ["PA"],#
    "AM": ["RR", "PA", "MT", "RO", "AC"],
    "BA": ["SE", "AL", "PE", "PI", "TO", "GO", "MG", "ES"],
    "CE": ["RN", "PB", "PE", "PI"],
    "DF": ["GO", "MG"],
    "ES": ["BA", "MG", "RJ"],
    "GO": ["BA", "TO", "MT", "MS", "MG", "DF"],
    "MA": ["PI", "TO", "PA"],
    "MT": ["AM", "PA", "TO", "GO", "MS", "RO"],
    "MS": ["MT", "GO", "MG", "SP", "PR"],
    "MG": ["BA", "GO", "MS", "SP", "RJ", "ES"],
    "PA": ["AP", "MA", "TO", "MT", "AM", "RR"],
    "PB": ["RN", "CE", "PE"],
    "PR": ["MS", "SP", "SC"],
    "PE": ["PB", "CE", "AL", "BA"],
    "PI": ["MA", "CE", "PE", "BA", "TO"],
    "RJ": ["ES", "MG", "SP"],
    "RN": ["CE", "PB"],
    "RS": ["SC"],
    "RO": ["AC", "AM", "MT"],
    "RR": ["AM", "PA"],
    "SC": ["PR", "RS"],
    "SP": ["RJ", "MG", "MS", "PR"],
    "SE": ["AL", "BA"],
    "TO": ["PA", "MA", "PI", "BA", "GO", "MT"]
}


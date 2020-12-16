import pandas as pd

def data_load(location):
    df = pd.read_csv('weatherAUS.csv')

    citiesStates = {'New South Wales': ['Albury','BadgerysCreek', 'Cobar' , 'CoffsHarbour' ,'Moree', 'Newcastle', 'NorahHead', 'Penrith', 'Sydney', 'SydneyAirport','WaggaWagga' , 'NorfolkIsland', 'Williamtown', 'Wollongong', 'MountGinini'], 
    'Victoria': ['Ballarat', 'Bendigo', 'Sale','Richmond' , 'MelbourneAirport', 'Melbourne', 'Mildura', 'Nhil', 'Portland', 'Watsonia' , 'Dartmoor'],
    'Queensland': ['Brisbane','Cairns', 'GoldCoast', 'Townsville'],
    'South Australia': ['Adelaide', 'MountGambier', 'Nuriootpa', 'Woomera'],
    'Western Australia': ['Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport', 'Perth', 'SalmonGums', 'Walpole'],
    'Tasmania': ['Hobart', 'Launceston'],
    'Australian Capital Territory': ['Canberra', 'Tuggeranong'],
    'Northern Territory': ['Katherine', 'Uluru', 'AliceSprings', 'Darwin']}

    if location != "all":
        
        cities = citiesStates[location]
        
        frames = []
        for c in cities:
            frames.append(df[df['Location'].str.contains(c)])
    
        df = pd.concat(frames)

    # To show how many not-null values each feature has
    # print(df.count().sort_values())

    # Deleting features that have, at least, 21% null values
    df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date'], axis='columns')

    # Drops lines that have at least 1 null value
    df = df.dropna(how='any')

    df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace = True)
    df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace = True)

    df.WindGustDir = pd.Categorical(pd.factorize(df.WindGustDir)[0])
    df.WindDir9am = pd.Categorical(pd.factorize(df.WindDir9am)[0])
    df.WindDir3pm = pd.Categorical(pd.factorize(df.WindDir3pm)[0])

    # Class count
    #count_class_0, count_class_1 = df['RainTomorrow'].value_counts()

    #print(count_class_0)
    #print(count_class_1)
    # Divide by class
    
    #df_class_0 = df.loc[df['RainTomorrow'] == 0]
    #df_class_1 = df.loc[df['RainTomorrow'] == 1]

    #df_class_1_over = df_class_1.sample(count_class_0, replace=True)
    #df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

    #df_class_0_under = df_class_0.sample(count_class_1)
    #df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

    #print('Random over-sampling:')
    #print(df_test_over['RainTomorrow'].value_counts())
    #print(df_test_under['RainTomorrow'].value_counts())
    #print(df)

    #print(df_test_over)

    #print(df.columns)

    return df
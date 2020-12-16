import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from scipy.stats import kruskal
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sns

def feature_selection_f_classif(x, y):
    k_best = SelectKBest(f_classif, k=10)
    fit = k_best.fit(x, y)
    #print("Scores: ", fit.scores_)
    ranking = fit.get_support()
    #print("Ranking: ", ranking)
    features = fit.transform(x)
    return features
    
def feature_selection_chi_squared(x, y):
    #Chi Squared
    # Feature extraction
    test = SelectKBest(score_func=chi2, k=10)
    fit = test.fit(x, y)

    # Summarize scores
    np.set_printoptions(precision=3)
    #print(fit.scores_)
    ranking = fit.get_support()
    #print("Ranking: ", ranking)

    features = fit.transform(x)
    # Summarize selected features
    return features

def takeSecond(elem):
    return elem[1]

def kruskal_wallis_against_target(x, y, n):
    h_list = []
    for i in range(x[0].size - 1):
        a = [value[i] for value in x]
        stat, p = kruskal(a, y)
        h_list.append((i, stat))
    highest_h = sorted(h_list, reverse=True, key=takeSecond)[:n]
    #print("\n\n\nKRUSKAL:", highest_h)
    final = []
    for line in x:
        aux = []
        for pos in highest_h:
            aux.append(line[pos[0]])
        final.append(aux)
    return final

def pearson_correlation_selection(df):
    #Using Pearson Correlation

    plt.figure(figsize=(12,10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap=plt.cm.Reds)
    #plt.show()

    #Correlation with output variable
    cor_target = abs(corr_matrix["RainTomorrow"])
    #Selecting highly correlated features
    relevant_features = cor_target[cor_target>.19]
    #print("\n\n RELEVANT ONES:\n:", relevant_features)

    #print(df[["Rainfall","WindGustSpeed","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Temp3pm","RainToday"]].corr())

    df = df[["Rainfall","WindGustSpeed","Humidity3pm","Pressure9am","RainToday", "Temp3pm", "RainTomorrow"]]

    return df

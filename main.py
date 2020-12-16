from classification import *
from data_load import *
from scaler import *
from feature_selection import *
from dimensionality_reduction import *

def data_load_menu():
    print("Escolha uma localidade")
    print("[1] All")
    print("[2] New South Wales")
    print("[3] Victoria")
    print("[4] Queensland")
    print("[5] Western Australia")
    print("[6] South Australia")
    print("[7] Tasmania")
    key = int(input())

    if key == 1:
        return data_load("all")
    elif key == 2:
        return data_load("New South Wales")
    elif key == 3:
        return data_load("Victoria")
    elif key == 4:
        return data_load("Queensland")
    elif key == 5:
        return data_load("Western Australia")
    elif key == 6:
        return data_load("South Australia")
    elif key == 7:
        return data_load("Tasmania")
    
    return null

def scaler(x):
    print("Scaler")
    print("[1] Min max scaler")
    print("[2] Standard scaler")
    print("[0] None")
    key = int(input())
    
    if key == 1:
        x = min_max_scaler(x) 
    elif key == 2:
        x = standard_scaler(x)
    elif key == 0:
        return x
    return x

def feature_selection(x, y, columns):
    print("Feature Selection")
    print("[1] f classif")
    print("[2] chi squared")
    print("[3] kruskal")
    print('[4] Pearson correlation selection')
    print("[0] None")
    key = int(input())
    
    if key == 1:
        x = feature_selection_f_classif(x, y)
    elif key == 2:
        x = feature_selection_chi_squared(x, y)
    elif key == 3:
        x = kruskal_wallis_against_target(x, y, 10)
    elif key == 4:
        df = pd.DataFrame(x, columns=columns)
        df = pd.concat([df, pd.DataFrame(y, columns=["RainTomorrow"])], axis=1)
        df = pearson_correlation_selection(df)
        
        x = df.loc[:, df.columns != "RainTomorrow"].values

    elif key == 0:
        return x
    return x
    
def reduction(x, y):
    print("Dimensionality Reduction")
    print("[1] PCA")
    print("[2] LDA")
    print("[0] None")
    key = int(input())

    if key == 1:
        x = pca2(x)
    elif key == 2:
        x = lda(x,y)
    elif key == 0:
        return x
    
    return x

def classification_models(x, y):
    print("Classification")
    print("[1] lda ")
    print("[2] Nearest centroid")
    print("[3] Random forest ")
    print("[4] Decision Tree")
    print("[5] GaussianNB")
    print("[6] KNeighbors")
    print("[7] SVC")


    key = int(input())

    if key == 1:
        linear_discriminant_analysis(x, y)
    elif key == 2:
        nearest_centroid(x, y)
    elif key == 3:
        random_forest(x, y)
    elif key == 4:
        decision_tree(x, y)
    elif key == 5:
        gaussian_NB(x, y)
    elif key == 6:
        k_neighbors(x, y)
    elif key == 7:
        svc(x, y)

if __name__ == '__main__':
    df = data_load_menu()

    # Dividing into features and target
    x = df.loc[:, df.columns != "RainTomorrow"].values
    y = df.RainTomorrow.values

    x = scaler(x)
    x = feature_selection(x, y, df.loc[:, df.columns != "RainTomorrow"].columns)
    x = reduction(x, y)
    classification_models(x, y)

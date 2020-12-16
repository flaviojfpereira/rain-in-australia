from classification import *
from data_load import *
from scaler import *
from feature_selection import *
from dimensionality_reduction import *
from statistics import *
import tkinter as tk
import copy



def data_load_menu(location):
    print("Escolha uma localidade")
    print("[1] All")
    print("[2] New South Wales")
    print("[3] Victoria")
    print("[4] Queensland")
    print("[5] Western Australia")
    print("[6] South Australia")
    print("[7] Tasmania")
    

    if location == 0:
        return data_load("all")
    elif location == 1:
        return data_load("New South Wales")
    elif location == 2:
        return data_load("Victoria")
    elif location == 3:
        return data_load("Queensland")
    elif location == 4:
        return data_load("Western Australia")
    elif location == 5:
        return data_load("South Australia")
    elif location == 6:
        return data_load("Tasmania")
    
    return

def select_scaler(x, scaler):
    print("Scaler")
    print("[1] Min max scaler")
    print("[2] Standard scaler")
    print("[0] None")
    
    if scaler == 0:
        x = min_max_scaler(x) 
    elif scaler == 1:
        x = standard_scaler(x)
    elif scaler == 2:
        return x
    return x

def feature_selection(x, y, columns, feature_selector):
    print("Feature Selection")
    print("[1] f classif")
    print("[2] chi squared")
    print("[3] kruskal")
    print('[4] Pearson correlation selection')
    print("[0] None")
    
    if feature_selector == 0:
        x = feature_selection_f_classif(x, y)
    elif feature_selector == 1:
        x = feature_selection_chi_squared(x, y)
    elif feature_selector == 2:
        x = kruskal_wallis_against_target(x, y, 10)
    elif feature_selector == 3:
        df = pd.DataFrame(x, columns=columns)
        df = pd.concat([df, pd.DataFrame(y, columns=["RainTomorrow"])], axis=1)
        df = pearson_correlation_selection(df)
        #Violin Plot to show some features' distributions
        data = df
        data = pd.melt(data,id_vars="RainTomorrow",
                            var_name="features",
                            value_name='value')
        plt.figure(figsize=(7,7))
        sns.violinplot(x="features", y="value", hue="RainTomorrow", data=data,split=True, inner="quart")
        plt.xticks(rotation=90)
        #plt.show()

        x = df.loc[:, df.columns != "RainTomorrow"].values

    elif feature_selector == 4:
        return x
    return x
    
def reduction(x, y, dimensionality_reductor):
    print("Dimensionality Reduction")
    print("[1] PCA")
    print("[2] LDA")
    print("[0] None")

    if dimensionality_reductor == 0:
        x = pca2(x)
    elif dimensionality_reductor == 1:
        x = lda(x,y)
    elif dimensionality_reductor == 2:
        return x
    
    return x

def classification_models(x, y, classifier):
    print("Classification")
    print("[1] lda ")
    print("[2] Nearest centroid")
    print("[3] Random forest ")
    print("[4] Decision Tree")
    print("[5] GaussianNB")
    print("[6] KNeighbors")
    print("[7] SVC")

    if classifier == 0:
        return linear_discriminant_analysis(x, y)
    elif classifier == 1:
        return nearest_centroid(x, y)
    elif classifier == 2:
        return random_forest(x, y)
    elif classifier == 3:
        return decision_tree(x, y)
    elif classifier == 4:
        return gaussian_NB(x, y)
    elif classifier == 5:
        return k_neighbors(x, y)
    elif classifier == 6:
        return svc(x, y)

def execute(v, v1, v2, v3, classifier, results_label):

    location = v.get()
    scaler = v1.get()
    feature_selector = v2.get()
    dimensionality_reductor = v3.get()
    classifier = classifier.get()
    print(location, scaler, feature_selector, dimensionality_reductor, classifier)


    df = data_load_menu(location)
    print('load')

    # Dividing into features and target
    x = df.loc[:, df.columns != "RainTomorrow"].values
    y = df.RainTomorrow.values

    x = select_scaler(x, scaler)
    print('scal')
    x = feature_selection(x, y, df.loc[:, df.columns != "RainTomorrow"].columns, feature_selector)
    print('feat')
    x = reduction(x, y, dimensionality_reductor)
    print('reduc')

    msg = classification_models(x, y, classifier)
    results_label['text'] = msg
        
    

def feature_assessment():
    df = data_load("all")

    #Standard scaling
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
    df = scaled_df

    x = df.loc[:, df.columns != "RainTomorrow"].values
    y = df.RainTomorrow.values

    all_columns_array = [df['MinTemp'].values, df['MaxTemp'].values, df['Rainfall'].values, df['WindGustDir'].values, df['WindGustSpeed'].values,
                        df['WindDir9am'].values, df['WindDir3pm'].values, df['WindSpeed9am'].values, df['WindSpeed3pm'].values, df['Humidity9am'].values, 
                        df['Humidity3pm'].values, df['Pressure9am'].values, df['Pressure3pm'].values, df['Temp9am'].values, df['Temp3pm'].values, 
                        df['RainToday'].values]
    #print(df.dtypes)

    print(all_columns_array)
    #NORMALITY TESTS
    normality_results = []
    for i in range(len(all_columns_array)):
        print(str(i)+' - ',all_columns_array[i])
        normality_results.append(test_normal_ks([all_columns_array[i]]))

    for i in range(len(normality_results)):
        print(df.columns[i]+' -> ', normality_results[i], '\n')


    plt.subplot(221)
    histogram_norm(all_columns_array[0], df.columns[0], "value", "quantity")

    plt.subplot(223)
    plt.boxplot(all_columns_array[0], labels=[''])

    plt.subplot(222)
    histogram_norm(all_columns_array[1], df.columns[1], "value", "quantity")

    plt.subplot(224)
    plt.boxplot(all_columns_array[1], labels=[''])

    plt.show()

    plt.subplot(221)
    histogram_norm(all_columns_array[2], df.columns[2], "value", "quantity")

    plt.subplot(223)
    plt.boxplot(all_columns_array[2], labels=[''])

    plt.subplot(222)
    histogram_norm(all_columns_array[3], df.columns[3], "value", "quantity")
    
    plt.subplot(224)
    plt.boxplot(all_columns_array[3], labels=[''])

    plt.show()

    plt.subplot(221)
    histogram_norm(all_columns_array[4], df.columns[4], "value", "quantity")

    plt.subplot(223)
    plt.boxplot(all_columns_array[4], labels=[''])

    plt.subplot(222)
    histogram_norm(all_columns_array[5], df.columns[5], "value", "quantity")

    plt.subplot(224)
    plt.boxplot(all_columns_array[5], labels=[''])

    plt.show()

    plt.subplot(221)
    histogram_norm(all_columns_array[6], df.columns[6], "value", "quantity")

    plt.subplot(223)
    plt.boxplot(all_columns_array[6], labels=[''])

    plt.subplot(222)
    histogram_norm(all_columns_array[7], df.columns[7], "value", "quantity")

    plt.subplot(224)
    plt.boxplot(all_columns_array[7], labels=[''])

    plt.show()

    plt.subplot(221)
    histogram_norm(all_columns_array[8], df.columns[8], "value", "quantity")

    plt.subplot(223)
    plt.boxplot(all_columns_array[8], labels=[''])

    plt.subplot(222)
    histogram_norm(all_columns_array[9], df.columns[9], "value", "quantity")

    plt.subplot(224)
    plt.boxplot(all_columns_array[9], labels=[''])

    plt.show()

    plt.subplot(221)
    histogram_norm(all_columns_array[10], df.columns[10], "value", "quantity")

    plt.subplot(223)
    plt.boxplot(all_columns_array[10], labels=[''])

    plt.subplot(222)
    histogram_norm(all_columns_array[11], df.columns[11], "value", "quantity")

    plt.subplot(224)
    plt.boxplot(all_columns_array[11], labels=[''])

    plt.show()
    
    plt.subplot(221)
    histogram_norm(all_columns_array[12], df.columns[12], "value", "quantity")

    plt.subplot(223)
    plt.boxplot(all_columns_array[12], labels=[''])

    plt.subplot(222)
    histogram_norm(all_columns_array[13], df.columns[13], "value", "quantity")

    plt.subplot(224)
    plt.boxplot(all_columns_array[13], labels=[''])

    plt.show()

    plt.subplot(221)
    histogram_norm(all_columns_array[14], df.columns[14], "value", "quantity")

    plt.subplot(223)
    plt.boxplot(all_columns_array[14], labels=[''])

    plt.subplot(222)
    histogram_norm(all_columns_array[15], df.columns[15], "value", "quantity")

    plt.subplot(224)
    plt.boxplot(all_columns_array[15], labels=[''])

    plt.show()


if __name__ == '__main__':
    ###################INTERFACE###################
    root = tk.Tk()
    root.title("Rain in Australia")
    root.geometry('1200x1200')

    v = tk.IntVar()
    v.set(0)  # initializing the choice, i.e. Python

    locations = [
        "All",
        "New South Wales",
        "Victoria",
        "Queensland",
        "Western Australia",
        "South Australia",
        "Tasmania"
    ]

    label = tk.Label(root, text="Location:", font=("Helvetica", 16), padx=20)
    label.pack(anchor=tk.W)

    relys = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    i = 0
    for val, language in enumerate(locations):
        tk.Radiobutton(root, 
                    text=language,
                    padx = 20,
                    pady = 5,
                    font=("Helvetica", 10),
                    variable=v, 
                    command=v.get(),
                    value=val).place(relx=0,rely=relys[i])
        i+=1

    v1 = tk.IntVar()
    v1.set(2)  # initializing the choice, i.e. Python

    scalers = [
        "MinMax Scaler",
        "Standard Scaler",
        "None"
    ]

    label = tk.Label(root, text="Scaler:", padx=20, font=("Helvetica", 16))
    label.place(rely = 0.40)

    relys = [0.45, 0.5, 0.55]
    i = 0
    for val, language in enumerate(scalers):
        tk.Radiobutton(root, 
                    text=language,
                    padx = 20,
                    pady = 5,
                    font=("Helvetica", 10),
                    variable=v1, 
                    command=v1.get(),
                    value=val).place(relx=0,rely=relys[i])
        i+=1

    v2 = tk.IntVar()
    v2.set(4)  # initializing the choice, i.e. Python

    feature_selectors = [
        "F Classif Statistical Test",
        "Chi Squared Test",
        "Kruskal Wallis",
        "Using Pearson's Correlation",
        "None"
    ]

    label = tk.Label(root, text="Feature Selector:", padx=20, font=("Helvetica", 16))
    label.place(rely = 0.60)

    relys = [0.65, 0.7, 0.75, 0.8, 0.85]
    i = 0
    for val, language in enumerate(feature_selectors):
        tk.Radiobutton(root, 
                    text=language,
                    padx = 20,
                    pady = 5,
                    font=("Helvetica", 10),
                    variable=v2, 
                    command=v2.get(),
                    value=val).place(relx=0,rely=relys[i])
        i+=1    

    v3 = tk.IntVar()
    v3.set(2)  # initializing the choice, i.e. Python

    dimensionality_reductors = [
        "PCA",
        "LDA",
        "None"
    ]

    label = tk.Label(root, text="Dimensionality Reduction:", padx=20, font=("Helvetica", 16))
    label.place(relx=0.2, rely=0)

    relys = [0.05, 0.1, 0.15]
    i = 0
    for val, language in enumerate(dimensionality_reductors):
        tk.Radiobutton(root, 
                    text=language,
                    padx = 20,
                    pady = 5,
                    font=("Helvetica", 10),
                    variable=v3, 
                    command=v3.get(),
                    value=val).place(relx=0.2,rely=relys[i])
        i+=1  

    classifier = tk.IntVar()
    classifier.set(0)  # initializing the choice, i.e. Python

    classifiers = [
        "LDA",
        "Nearest Centroid",
        "Random Forest",
        "Decision Tree",
        "GaussianNB",
        "KNeighbours",
        "SVC"
    ]

    label = tk.Label(root, text="Classifier:", padx=20, font=("Helvetica", 16))
    label.place(relx=0.2, rely=0.2)

    relys = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    i = 0
    for val, language in enumerate(classifiers):
        tk.Radiobutton(root, 
                    text=language,
                    padx = 20,
                    pady = 5,
                    font=("Helvetica", 10),
                    variable=classifier, 
                    command=classifier.get(),
                    value=val).place(relx=0.2,rely=relys[i])
        i+=1  

    results_frame = tk.Frame(root, bg='grey', bd = 10)
    results_frame.place(relx=0.45, rely=0.1, relwidth=0.5, relheight=0.8)

    results_label = tk.Label(results_frame, font=("Helvetica", 16) )
    results_label.place(relwidth=1,relheight=1)

    run_button = tk.Button(root, text="Run program!", command=lambda : execute(v, v1, v2, v3, classifier, results_label)) 
    run_button.place(relx=0.225, rely=0.85, relwidth=0.1, relheight=0.05)

    feature_analysis_but = tk.Button(root, text="Give me\nthe stats!", command=lambda : feature_assessment()) 
    feature_analysis_but.place(relx=0.283, rely=0.70, relwidth=0.05, relheight=0.06)

    feature_analysis_label = tk.Label(root, text="Click here for Feature Assessment", font=("Helvetica", 10, "bold") )
    feature_analysis_label.place(relx=0.225,rely=0.65)

    feature_analysis_label2 = tk.Label(root, text="and Visualization!", font=("Helvetica", 10, "bold") )
    feature_analysis_label2.place(relx=0.225,rely=0.675)

    root.mainloop()
    ###############END INTERFACE##################


    
    
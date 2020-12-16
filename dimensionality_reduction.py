from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd


def pca(x):
    pca = PCA(n_components=4)
    fit = pca.fit(x)
    # summarize components
    #print(fit.components_)
    return fit.components_

def pca2(x):
    # pca - keep 95% of variance
    pca = PCA(0.95)

    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(data = principal_components)
    #print("\n\nPCA SHAPE: " , principal_df.shape)
    return principal_df

def lda(x, y): #investigate more about LDA, it can be a model or a feature selector
    lda = LinearDiscriminantAnalysis(n_components=8)
    x = lda.fit(x, y).transform(x)
    ## View Percentage Of Variance Retained By New Features
    ## View the ratio of explained variance
    #print(lda.explained_variance_ratio_)
    return x


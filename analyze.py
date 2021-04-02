import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import shap
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def shap_analysis(X, y, features, target):
    kf = KFold(n_splits=10, shuffle=True)

    rf = ExtraTreesRegressor(n_estimators=500)
    print (target, cross_val_score(rf, X, y, cv=kf, scoring='r2').mean())
    # rf.fit(X, y)
    # explainer = shap.KernelExplainer(rf.predict, X)
    # shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values, features=X, feature_names=features, title=target)

    # kernel = ConstantKernel() * RBF() + WhiteKernel()
    # gp = GaussianProcessRegressor(kernel=kernel)
    # print (target, cross_val_score(gp, X, y, cv=kf, scoring='r2').mean())
    # gp.fit(X[train], y[train])
    # explainer = shap.KernelExplainer(gp.predict, X)
    # shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values, features=X, feature_names=features, title=target)

    # svr = SVR()
    # print (target, cross_val_score(svr, X, y, cv=kf, scoring='r2').mean())
    # svr.fit(X, y)
    # explainer = shap.KernelExplainer(svr.predict, X)
    # shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values, features=X, feature_names=features, title=target)



df = pd.read_csv('all.csv')
features = ['HAuCl4', 'PVP', 'Glucose', 'NaOH', 'Time']
targets = ['A450', 'R', 'FWHM', 'WL']

scaler = StandardScaler()
X_raw = np.array(df.loc[:, features])
for target in targets:
    y_raw = np.array(df.loc[:, [target]])
    condition = np.logical_or(y_raw == 0, y_raw == 470)
    removed = np.where(condition)
    y = np.delete(y_raw, removed)
    X = np.delete(X_raw, removed, axis=0)
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    scaler.fit(y.reshape(-1,1))
    y_scaled = scaler.transform(y.reshape(-1,1)).ravel()
    # print (X, y)
    shap_analysis(X_scaled, y_scaled, features, target)


# scaler.fit(X_raw)
# X_scaled = scaler.transform(X_raw)
# pca_all = PCA()
# pca_all.fit(X_scaled)
# X_pca_all = pca_all.transform(X_scaled)
# print (pca_all.explained_variance_ratio_ * 100)

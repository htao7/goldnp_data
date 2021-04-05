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

    # model = ExtraTreesRegressor(n_estimators=500)

    # kernel = ConstantKernel() * RBF() + WhiteKernel()
    # model = GaussianProcessRegressor(kernel=kernel)

    model = SVR()

    print (target, cross_val_score(model, X, y, cv=kf, scoring='r2').mean())


    clf = model.fit(X, y)
    print(target, 'non-CV', clf.score(X, y))
    y_pred = model.predict(X=X)
    # print (y_pred - y)
    # plt.scatter(np.arange(len(y)), y_pred - y)
    # plt.title(target)
    # plt.xlabel('exp #')
    # plt.ylabel('residual')
    # plt.show()
    # explainer = shap.KernelExplainer(model.predict, X)
    # shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values, features=X, feature_names=features, title=target)




df = pd.read_csv('all.csv')
features = ['HAuCl4', 'PVP', 'Glucose', 'NaOH', 'Time']
targets = ['A450', 'R', 'FWHM', 'WL']

scaler = StandardScaler()
X_raw = np.array(df.loc[:, features])
# X_explor = [X_raw[2 * i + 1] for i in range(int(len(X_raw) / 2))]
for target in targets:
    y_raw = np.array(df.loc[:, [target]])
    # y_explor = [y_raw[2 * i + 1] for i in range(int(len(y_raw) / 2))]
    # condition = np.logical_or(y_explor == 0, y_explor == 470)
    # removed = np.where(condition)
    # y = np.delete(y_explor, removed)
    # X = np.delete(X_explor, removed, axis=0)

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

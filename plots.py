#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import json
import matplotlib
import seaborn as sns
import os
import argparse
import sys
from itertools import combinations

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
# import shap

colors = ['#4ecdc4', '#dce2dc', '#ff6b6b', '#1a535c', '#ffe66d']
# modifier to add to 0 in logplot
LOG_MODIFIER = 1e-2

VMIN = None
VMAX = None

plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)

# =============
# Parse Options
# =============
def parse_options():

    parser = argparse.ArgumentParser()

    parser.add_argument('-f',
                        dest='file',
                        type=str,
                        help='Excel or CSV file.')
    parser.add_argument('-t',
                        dest='targets',
                        type=str,
                        help='List of column headers with objectives.',
                        nargs='+')
    parser.add_argument('--absolutes',
                        metavar="KEY=VALUE",
                        dest='absolutes',
                        help='List of absolute threshold for each objectives. E.g. "obj1=10 obj2=0.1".',
                        nargs='+')
    parser.add_argument('--logplot',
                        dest='logplot',
                        default=False,
                        help='Whether to plot the logarithm for the objectives.',
                        action='store_true')
    parser.add_argument('--feat1',
                        dest='feat1',
                        type=str,
                        help='Feature that you want to plot against the targets.',
                        default=None)
    parser.add_argument('--feat2',
                        dest='feat2',
                        type=str,
                        help='Second feature that you want to plot against the targets.',
                        default=None)
    parser.add_argument('--model',
                        dest='model_type',
                        type=str,
                        help='Second feature that you want to plot against the targets.',
                        default='gp',
                        choices=['gp', 'rf', 'svm'])
    parser.add_argument('--ncountours',
                        dest='ncountours',
                        type=int,
                        help='Number of contour lines to plot.',
                        default=5)
    parser.add_argument('--goals',
                        metavar="KEY=VALUE",
                        dest='goals',
                        help='List of "min" or "max" goals for the objectives. E.g. "obj1=min obj2=max".',
                        nargs='+')
    parser.add_argument('--pareto_cmap',
                        dest='pareto_cmap',
                        type=str,
                        help='Whether to show a color gradient for the scatter plot markers in the plot showing '
                             'multiple objectives. Options are: "none" to not use a color gradient, "all" to use a '
                             'gradient for all points, and "exploit" ro use it for the exploitation points only.',
                        default='none',
                        choices=['none', 'all', 'exploit'])

    args = parser.parse_args()
    args.absolutes = parse_vars(args.absolutes)
    args.goals = parse_vars(args.goals)
    return args


# ==================
# Plotting Functions
# ==================
def plot_trace(df, goals, logplot=False, absolutes=None):
    best_data = {}
    targets = [c for c in df.columns]

    for target in targets:
        best_data[target] = []
        x = np.array(df.loc[:, target])
        for i, xi in enumerate(x):
            if goals[target] == 'max':
                best_data[target].append(np.max(x[:i + 1]))
            else:
                best_data[target].append(np.min(x[:i + 1]))

    # plot
    nrows = 1
    ncols = len(targets)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 5))
    if logplot is True:
        fig.suptitle('Best result vs. number or experiments, y_Logscale')
    else:
        fig.suptitle('Best result vs. number or experiments')

    # put single axis in list
    if isinstance(axs, (plt.Axes, dict)):
        axs = [axs]

    i = 0
    for target in best_data:
        x = range(1, len(best_data[target]) + 1)
        if logplot is True:
            if 0.0 in best_data[target]:
                y = np.log10(np.array(best_data[target]) + LOG_MODIFIER)
            else:
                y = np.log10(best_data[target])
        else:
            y = best_data[target]

        axs[i].plot(x, y, linewidth=5, color='k')
        axs[i].plot(x, y, linewidth=4, color=colors[i])
        axs[i].set_xlabel('# experiments')
        axs[i].set_ylabel(target)
        axs[i].grid(linestyle=':')

        # absolutes
        if absolutes is not None:
            if target in absolutes:
                threshold = absolutes[target]
                if logplot is True:
                    if threshold == 0.0:
                        threshold = np.log10(threshold + LOG_MODIFIER)
                    else:
                        threshold = np.log10(threshold)
                axs[i].axhline(y=threshold, linestyle='--', color='k', linewidth=2, zorder=0)

        i += 1


def plot_one_contour(ax, df, target, logplot, features, model_type, n_countours=5):
    x0_name = features[0]
    x1_name = features[1]

    X_raw = np.array(df.loc[:, features])
    y_raw = np.array(df.loc[:, [target]])

    condition = np.logical_or(y_raw == 0, y_raw == 470)
    removed = np.where(condition)
    y = np.delete(y_raw, removed)
    X = np.delete(X_raw, removed, axis=0)

    # fit model
    feat_scaler = StandardScaler()
    targ_scaler = StandardScaler()
    feat_scaler.fit(X)
    targ_scaler.fit(y.reshape(-1,1))

    X_scaled = feat_scaler.transform(X)
    y_scaled = targ_scaler.transform(y.reshape(-1,1)).ravel()

    if model_type == 'rf':
        model = ExtraTreesRegressor(n_estimators=500)
    elif model_type == 'gp':
        kernel = ConstantKernel() * RBF() + WhiteKernel()
        model = GaussianProcessRegressor(kernel=kernel)
    elif model_type == 'svm':
        model = SVR()
    else:
        raise ValueError('cannot recognized chosen model type')

    model.fit(X=X_scaled, y=y_scaled)


    # get data ranges
    dfmins = np.min(df.loc[:, features], axis=0)
    dfmaxs = np.max(df.loc[:, features], axis=0)
    df_ranges = pd.concat([dfmins, dfmaxs], axis=1)
    df_ranges.columns = ['min', 'max']

    # define limits
    x0_lims = [df_ranges.loc[x0_name, 'min'], df_ranges.loc[x0_name, 'max']]
    x1_lims = [df_ranges.loc[x1_name, 'min'], df_ranges.loc[x1_name, 'max']]


    # data
    N = 100

    x0 = np.linspace(x0_lims[0], x0_lims[1], N)
    x1 = np.linspace(x1_lims[0], x1_lims[1], N)
    X0, X1 = np.meshgrid(x0, x1)
    df_predict = pd.DataFrame({x0_name: X0.flatten(), x1_name: X1.flatten()})

    # predict
    df_predict_scaled = feat_scaler.transform(df_predict)
    y_pred_scaled = model.predict(X=df_predict_scaled)
    y_pred = targ_scaler.inverse_transform(y_pred_scaled)

    if logplot is True:
        if 0.0 in y_pred:
            y_pred = y_pred + LOG_MODIFIER
        y_pred = np.log10(y_pred)

    # plot countour plot
    contours = ax.contour(X0, X1, np.reshape(y_pred, newshape=X0.shape), n_countours, colors='black')
    _ = ax.clabel(contours, inline=True, fontsize=12, fmt='%.1f')

    vmin = VMIN
    vmax = VMAX
    if vmin is None:
        vmin = np.min(y_pred)
    if vmax is None:
        vmax = np.max(y_pred)

    m = ax.imshow(np.reshape(y_pred, newshape=X0.shape), extent=[x0_lims[0], x0_lims[1], x1_lims[0], x1_lims[1]],
                  origin='lower', cmap='RdGy', alpha=0.5, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(mappable=m, ax=ax, shrink=0.7)
    cbar.set_label(target)

    # scatter plot
    x0_samples = np.array(df.loc[:, [x0_name]])
    x1_samples = np.array(df.loc[:, [x1_name]])

    ax.scatter(x0_samples, x1_samples, s=100, marker='X', color=colors[4], edgecolor='k', label='Max', linewidth=1,
               zorder=10)

    # ax.scatter(0.2, 0.2, s=200, marker='X', color=colors[4], edgecolor='k', label='Max', linewidth=1,
    #            zorder=10)
    # ax.scatter(0.8, 0.2, s=200, marker='X', color=colors[4], edgecolor='k', label='Max', linewidth=1,
    #            zorder=10)
    # ax.scatter(0.8, 1.0, s=200, marker='X', color=colors[4], edgecolor='k', label='Max', linewidth=1,
    #            zorder=10)
    # ax.scatter(0.2, 0.7, s=200, marker='X', color=colors[4], edgecolor='k', label='Max', linewidth=1,
    #            zorder=10)
    # ax.scatter(0.9, 0.1, s=200, marker='X', color=colors[4], edgecolor='k', label='Max', linewidth=1,
    #            zorder=10)
    # ax.scatter(0.5, 0.5, s=200, marker='X', color=colors[4], edgecolor='k', label='Max', linewidth=1,
    #            zorder=10)
    # ax.scatter(0.3, 0.6, s=200, marker='X', color=colors[4], edgecolor='k', label='Max', linewidth=1,
    #            zorder=10)
    # ax.scatter(0.6, 0.8, s=200, marker='X', color=colors[4], edgecolor='k', label='Max', linewidth=1,
    #            zorder=10)
    # ax.scatter(0.9, 0.3, s=200, marker='X', color=colors[4], edgecolor='k', label='Max', linewidth=1,
    #            zorder=10)
    # ax.scatter(0.7, 0.3, s=200, marker='X', color=colors[4], edgecolor='k', label='Max', linewidth=1,
    #            zorder=10)
    # ax.scatter(0.2, 0.8, s=200, marker='X', color=colors[4], edgecolor='k', label='Max', linewidth=1,
    #            zorder=10)
    # ax.scatter(0.1, 0.1, s=200, marker='X', color=colors[4], edgecolor='k', label='Max', linewidth=1,
    #            zorder=10)
    # ax.scatter(0.9, 0.2, s=250, marker='X', color=colors[0], edgecolor='k', label='Max', linewidth=1,
    #            zorder=10)




    # labels
    ax.set_xlabel(x0_name)
    ax.set_ylabel(x1_name)

    # lims
    _ = ax.set_xlim(x0_lims)
    _ = ax.set_ylim(x1_lims)

    # title
    _ = ax.set_title(target)


def plot_contours(df, targets, features, model_type, n_countours=5, logplot=False):
    # plot
    nrows = 1
    ncols = len(targets)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 5))
    if logplot is True:
        fig.suptitle('z_Logscale')

    # put single axis in list
    if isinstance(axs, (plt.Axes, dict)):
        axs = [axs]

    for i, target in enumerate(targets):
        # plot_one_contour(ax=axs[i], df=df, target=target, features=features, model_type=model_type,
        #                  n_countours=n_countours)
        plot_one_contour(ax=axs[i], df=df, target=target, logplot=logplot, features=features, model_type=model_type,
                         n_countours=n_countours)


def plot_1d_fit(ax, df, target, logplot, feature, model_type, color):
    X_raw = np.array(df.loc[:, [feature]])
    y_raw = np.array(df.loc[:, [target]])

    condition = np.logical_or(y_raw == 0, y_raw == 470)
    removed = np.where(condition)
    y = np.delete(y_raw, removed, axis=0)
    X = np.delete(X_raw, removed, axis=0)

    # fit model
    feat_scaler = StandardScaler()
    targ_scaler = StandardScaler()
    feat_scaler.fit(X)
    targ_scaler.fit(y)

    X_scaled = feat_scaler.transform(X)
    y_scaled = targ_scaler.transform(y).ravel()

    if model_type == 'rf':
        model = ExtraTreesRegressor(n_estimators=500)
    elif model_type == 'gp':
        kernel = ConstantKernel() * RBF() + WhiteKernel()
        model = GaussianProcessRegressor(kernel=kernel)
    elif model_type == 'svm':
        model = SVR()
    else:
        raise ValueError('cannot recognized chosen model type')

    model.fit(X=X_scaled, y=y_scaled)

    # define limits
    x0_lims = [np.min(X), np.max(X)]

    #manual limits
    # x_lims = [2.2, 12]
    # y_lims = [90., 180.]

    # data
    N = 1000
    x0 = np.linspace(x0_lims[0], x0_lims[1], N)

    # predict
    df_predict_scaled = feat_scaler.transform(x0.reshape(-1, 1))

    if model_type == 'gp':
        y_pred_scaled, y_std_scaled = model.predict(X=df_predict_scaled, return_std=True)
        y_std = targ_scaler.inverse_transform(y_std_scaled)
        # print(y_std)
    else:
        y_pred_scaled = model.predict(X=df_predict_scaled)

    y_pred = targ_scaler.inverse_transform(y_pred_scaled)

    # std if gp
    if model_type == 'gp':
        lowerbound = y_pred - y_std
        upperbound = y_pred + y_std
        if logplot is True:
            lowerbound[lowerbound <= 0] = LOG_MODIFIER
            lowerbound = np.log10(lowerbound)
            if 0.0 in upperbound:
                upperbound = upperbound + LOG_MODIFIER
            upperbound = np.log10(upperbound)
        # ax.fill_between(x=x0, y1=lowerbound, y2=upperbound, color=color, alpha=0.3, label='uncertainty')

    if logplot is True:
        if 0.0 in y_pred:
            y_pred = y_pred + LOG_MODIFIER
        y_pred = np.log10(y_pred)


    # plot
    ax.scatter(X, y, s=100, marker='X', color=colors[4], edgecolor='k', label='samples', linewidth=1, zorder=0)
    ax.plot(x0, y_pred, linewidth=5, color='k')
    ax.plot(x0, y_pred, linewidth=4, color=color, label=f'{model_type} estimate')
    ax.grid(linestyle=':')

    # scatter plot
    # obs_x = np.array(df.loc[:, [feature]])
    # obs_y = np.array(df.loc[:, [target]])
    # ax.scatter(obs_x, obs_y, s=100, marker='X', color=colors[4], edgecolor='k', label='samples', linewidth=1, zorder=10)

    if logplot is True:
        if 0.0 in obs_y:
            obs_y = obs_y + LOG_MODIFIER
        obs_y = np.log10(obs_y)

    # labels and lims
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    _ = ax.set_xlim(x0_lims)

    #_ = ax.set_xlim(x_lims)
    #_ = ax.set_ylim(y_lims)

    # title
    _ = ax.set_title(target)
    #_ = ax.legend()


def plot_1d_fits(df, targets, feature, model_type, logplot=False):
    # plot
    nrows = 1
    ncols = len(targets)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 5))
    if logplot is True:
        fig.suptitle('y_Logscale')

    # put single axis in list
    if isinstance(axs, (plt.Axes, dict)):
        axs = [axs]

    for i, target in enumerate(targets):
        plot_1d_fit(ax=axs[i], df=df, target=target, logplot=logplot, feature=feature, model_type=model_type, color=colors[i])


def plot_pareto(df, targets, logplot=False, absolutes=None, goals=None, color_gradient=False, highlight_exploit=False):
    # get pairwise target combinations
    target_pairs = list(combinations(targets, 2))

    # plot
    nrows = 1
    ncols = len(target_pairs)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 5))
    if logplot is True:
        fig.suptitle('xy_Logscale')

    # put single axis in list
    if isinstance(axs, (plt.Axes, dict)):
        axs = [axs]

    for i, target_pair in enumerate(target_pairs):
        t1 = target_pair[0]
        t2 = target_pair[1]
        obs_x = np.array(df.loc[:, [t1]])
        obs_y = np.array(df.loc[:, [t2]])

        if logplot is True:
            if 0.0 in obs_x:
                obs_x = obs_x + LOG_MODIFIER
            obs_x = np.log10(obs_x)
            if 0.0 in obs_y:
                obs_y = obs_y + LOG_MODIFIER
            obs_y = np.log10(obs_y)

        # plot scatter
        if color_gradient is True:
            cmap = matplotlib.cm.get_cmap('YlOrBr')
            _colors = cmap(np.linspace(0, 1, len(obs_x)))
            for j, (ox, oy) in enumerate(zip(obs_x, obs_y)):
                if highlight_exploit is True:
                    expl = j % 2
                    if expl == 0:
                        axs[i].scatter(ox, oy, s=100, marker='X', color=_colors[j], edgecolor='k', label='samples',
                                       linewidth=1, zorder=10)
                    else:
                        axs[i].scatter(ox, oy, s=100, marker='X', color='white', edgecolor='k', label='samples',
                                       linewidth=1, zorder=10, alpha=0.4)
                else:
                    axs[i].scatter(ox, oy, s=100, marker='X', color=_colors[j], edgecolor='k', label='samples',
                                   linewidth=1, zorder=10)
            # add colorbar
            fig.subplots_adjust(right=0.8)
            cax = fig.add_axes([0.85, 0.3, 0.02, 0.4])
            norm = matplotlib.colors.Normalize(vmin=1, vmax=len(obs_x))
            cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
            cbar.set_label('# experiment')
        else:
            axs[i].scatter(obs_x, obs_y, s=100, marker='X', color=colors[4], edgecolor='k', label='samples',
                           linewidth=1, zorder=10)

        # absolutes
        if absolutes is not None:
            if t1 in absolutes:
                threshold1 = absolutes[t1]
                if logplot is True:
                    if threshold1 == 0.0:
                        threshold1 = np.log10(threshold1 + LOG_MODIFIER)
                    else:
                        threshold1 = np.log10(threshold1)
                axs[i].axvline(x=threshold1, linestyle='--', color='k', linewidth=2, zorder=0)

            if t2 in absolutes:
                threshold2 = absolutes[t2]
                if logplot is True:
                    if threshold2 == 0.0:
                        threshold2 = np.log10(threshold2 + LOG_MODIFIER)
                    else:
                        threshold2 = np.log10(threshold2)
                axs[i].axhline(y=threshold2, linestyle='--', color='k', linewidth=2, zorder=0)

            # goals / shade area
            if goals is not None:
                xlim = axs[i].get_xlim()
                ylim = axs[i].get_ylim()

                if t1 in goals:
                    goal = goals[t1]
                    if goal == 'min':
                        axs[i].fill_betweenx(y=ylim, x1=threshold1, x2=xlim[0], color='k', alpha=0.1)
                    elif goal == 'max':
                        axs[i].fill_betweenx(y=ylim, x1=threshold1, x2=xlim[1], color='k', alpha=0.1)
                    else:
                        raise ValueError()

                if t2 in goals:
                    goal = goals[t2]
                    if goal == 'min':
                        axs[i].fill_between(x=xlim, y1=threshold2, y2=ylim[0], color='k', alpha=0.1)
                    elif goal == 'max':
                        axs[i].fill_between(x=xlim, y1=threshold2, y2=ylim[1], color='k', alpha=0.1)
                    else:
                        raise ValueError()

                _ = axs[i].set_xlim(xlim)
                _ = axs[i].set_ylim(ylim)

        # labels and lims
        axs[i].set_xlabel(t1)
        axs[i].set_ylabel(t2)

        # title
        _ = axs[i].set_title(f'{t1} vs {t2}')


# ================
# Helper Functions
# ================
def parse_var(s):
    """
    Parse a key, value pair, separated by '='
    That's the reverse of ShellArgs.

    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    """
    items = s.split('=')
    key = items[0].strip() # we remove blanks around keys, as is logical
    if len(items) > 1:
        # rejoin the rest:
        try:
            value = float('='.join(items[1:]))
        except:
            value = '='.join(items[1:])
    return (key, value)


def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}

    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d


# ====
# Main
# ====
def main(args):
    # read data in
    if args.file.split('.')[-1] == 'csv':
        df = pd.read_csv(args.file)
    elif args.file.split('.')[-1] in ('xlsx', 'xls'):
        df = pd.read_excel(args.file)
    else:
        raise ValueError(f'cannot recognise file format of input {args.file}')




    # plot traces
    # plot_trace(df=df.loc[:, args.targets], goals=args.goals, logplot=args.logplot, absolutes=args.absolutes)

    # plot contours
    if args.feat1 is not None and args.feat2 is not None:
        plot_contours(df=df, targets=args.targets, features=[args.feat1, args.feat2],
                      model_type=args.model_type, n_countours=args.ncountours, logplot=args.logplot)
    elif args.feat1 is not None and args.feat2 is None:
        plot_1d_fits(df=df, targets=args.targets, feature=args.feat1, model_type=args.model_type, logplot=args.logplot)

    # plot pareto
    if len(args.targets) == 2:
        if args.pareto_cmap == 'none':
            color_gradient = False
            highlight_exploit = False
        elif args.pareto_cmap == 'all':
            color_gradient = True
            highlight_exploit = False
        elif args.pareto_cmap == 'exploit':
            color_gradient = True
            highlight_exploit = True
        else:
            color_gradient = False
            highlight_exploit = False
        plot_pareto(df=df, targets=args.targets, logplot=args.logplot, absolutes=args.absolutes, goals=args.goals,
                    color_gradient=color_gradient, highlight_exploit=highlight_exploit)

    plt.show()


if __name__ == '__main__':
    args = parse_options()
    # print (args)
    main(args)

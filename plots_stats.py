import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
import os, shutil
from statistics import median
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def filter(df):
    return df[(df["size"] > 5) & (df["density"] > .5)]

def outliers(df):
    print("n_data: {0}".format(len(df["size"])))
    Q3 = df.quantile(.75) ; Q1 = df.quantile(.25)
    IQR = Q3 - Q1
    return ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()

#1 - Figure 1-g
'''
bin_dist = [2,2,4,5,6,8,9,9,10]
e_max = [item for item in bin_dist if item >= .9*max(bin_dist)]
print(len(bin_dist) - len(e_max))
numb = len(bin_dist) - len(e_max)
print(e_max)
bin_dist.sort()
plt.bar(np.arange(0,numb),bin_dist[:numb],color = ['lightskyblue'])
plt.bar(np.arange(numb,9),bin_dist[numb:],color = ['lightcoral'])
plt.axhline(y=.9*max(bin_dist), color='dimgrey', linestyle='--')
plt.ylabel("Sum of synaptic inputs (h)")
plt.xlabel("Neuron index (j)")
plt.text(.36,9.1, "E%-winners-take-all")
plt.xticks(np.arange(0,10,2))
plt.show()
'''

#2 - Figure 3-b
'''
def plot_function(ax, beta, beta_float,ticks):
    df = pd.read_csv("k_winners\k_winners_{0}".format(beta))
    iter = np.arange(1,101)
    ax.bar(iter, df["new"], color = "lightskyblue")
    ax.set_title(r'$\bf\beta = {0}$'.format(beta_float))
    ax.set_xlim(0,ticks)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$|N_t|$')
    x_plot = list(df["new"]).index(0)
    print(x_plot)
    ax.axvline(x = x_plot+1, color = "dimgrey", ls = "--", lw = 1)
    ax2 = ax.twinx()
    ax2.set_ylabel(r'$\left|X_t\right|$')
    ax2.plot(iter,df["new_1"], color = "lightcoral", lw = .5)

fig, axes = plt.subplots(1,1)

beta = ["0_1","0_01","0_001"]
beta_float = [0.1,0.01,0.001]
ticks = [12,45,70]

plot_function(axes,beta[1],beta_float[1],ticks[1])

plt.show()
'''

#3 - Figure 3-c
'''
beta = ["1","01","001"]
beta_n =[.1,.01,.001]
j = 79
fig, ax  = plt.subplots(3,1,sharex=True)

def plot_axis(ax,data,beta_n,y):
    delta_fire = []
    for i in range(0,200):
        if i == 0:
            delta_fire.append(data.iloc[i])
        else:
            delta_fire.append(data.iloc[i] - data.iloc[i-1])

    ax.plot(np.arange(0,200),delta_fire, linewidth = 1, c = "lightcoral")
    ax.set_xlim(-1,200)
    #ax.set_ylim(-50,50)
    ax.set_ylabel(r'$\sum \Delta f$')
    ax.text(165,y,r'$\beta = {0}$'.format(beta_n))
    #ax.set_ylabel(r'$\sum\Delta f_{t}$')
y = [43,18,20]

for i in range(0,len(beta)):
    df = pd.read_csv("convergence\convergence_0_"+ beta[i] + "_" + str(j + 1))
    df_fire = df["fire"]
    plot_axis(ax[i],df_fire,beta_n[i],y[i])
print(df)
ax[2].set_xlabel(r'$t$')
plt.show()
'''

#4 - Figure 3-d
'''
size = 7
beta = ["1","05","01","005","001"]
inhibition = ["02","04","06","08","1"]
label_inhibition = ["- 0.2","- 0.4","- 0.6","- 0.8","- 1.0"]
data_inhibition = pd.DataFrame()
for j in range(0,5):
    data_list = []
    for i in beta:
        df = filter(pd.read_csv("{0}ormation_{3}\{1}ormation_0_{2}".format("f","f",i,inhibition[j])))
        data_list.append(len(df["density"])/500)
    print(data_list)
    data_inhibition[label_inhibition[j]] = data_list
print(data_inhibition)
ax = data_inhibition.plot(style='--o')
ax.legend(title = r'$\omega_{i}$', frameon = False)
ax.set_xticks(np.arange(0,5))
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylabel(r'Normalized success rate')
ax.set_xlabel(r'Synaptic plasticity $(\beta)$')
ax.set_xticklabels(["0.1","0.05","0.01","0.005","0.001"])

plt.show()
'''

#5 - Figure 4-a
'''
beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]
f, axes = plt.subplots(1,1)
conc_mdf = []

for i in range(0,5):
    df_emax = filter(pd.read_csv("{0}ormation_02\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0).assign(Trial = 0))
    df_kwin = pd.read_csv("{0}ormation_without_emax\{1}ormation_0_{2}_k_winners_".format("f","f",beta[i]), index_col=0).assign(Trial = 1)
    print("BETA: ", beta[i])
    print("===EMAX===")
    print(df_emax)
    print("===KWIN===")
    print(df_kwin)

    cdf = pd.concat([df_emax,df_kwin])
    mdf = pd.melt(cdf, id_vars=["time"], value_vars=["Trial"])
    mdf["variable"] = beta_float[i]
    conc_mdf.append(mdf)
cdf = pd.concat(conc_mdf)
cdf['value'] = np.where(cdf['value']==1, 'Original model', 'Adapted model')
sns.boxplot(x = "variable", hue = "value", y="time", data=cdf, flierprops={'marker': 'x', 'markersize': 5}, width=.7, palette=['lightskyblue', 'lightcoral'], linewidth=.5)
axes.legend(frameon=False)
axes.set_xlabel(r'Synaptic plasticity $(\beta)$')
axes.set_ylabel("Number of iterations (T)")
axes.spines[['right', 'top']].set_visible(False)

conc_mdf = []
for i in range(0,4):
    df_emax = filter(pd.read_csv("{0}ormation_02\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0).assign(Trial = 0))
    df_kwin = pd.read_csv("{0}ormation_without_emax\{1}ormation_0_{2}_k_winners_".format("f","f",beta[i]), index_col=0).assign(Trial = 1)
    print("BETA: ", beta[i])
    print("===EMAX===")
    print(df_emax)
    print("===KWIN===")
    print(df_kwin)

    cdf = pd.concat([df_emax,df_kwin])
    mdf = pd.melt(cdf, id_vars=["time"], value_vars=["Trial"])
    mdf["variable"] = beta_float[i]
    conc_mdf.append(mdf)
cdf = pd.concat(conc_mdf)
rect = (0.3, 0.35, 0.55, 0.4)
axin = f.add_axes(rect)
sns.boxplot(x = "variable", hue = "value", y="time", data=cdf, flierprops={'marker': 'x', 'markersize': 5}, width=.7, palette=['lightskyblue', 'lightcoral'], linewidth=.5,ax=axin)
axin.legend().remove()
axin.set_xlabel("")
axin.set_ylabel("")
axin.spines[['right', 'top']].set_visible(False)
plt.show()
'''

#6 - Figure 4-b
'''
def outliers(df):
    print("n_data: {0}".format(len(df["size"])))
    Q3 = df.quantile(.75) ; Q1 = df.quantile(.25)
    IQR = Q3 - Q1
    return ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()


beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]
f, axes = plt.subplots(1,2)
size = 7
conc_mdf = []
conc_02 = []
for i in range(0,5):
    df_without = filter(pd.read_csv("{0}ormation_without\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0).assign(Trial = 0))
    df_with = filter(pd.read_csv("{0}ormation_02\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0).assign(Trial = 1))
    df_02 = filter(pd.read_csv("{0}ormation_02\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0).assign(Trial = beta_float[i]))
    conc_02.append(df_02)
    print("BETA: ", beta[i])
    print("========WITHOUT========")
    print(df_without.describe())
    print("========WITH========")
    print(df_with.describe())
    print()

    cdf = pd.concat([df_without,df_with])
    mdf = pd.melt(cdf, id_vars=["size"], value_vars=["Trial"])
    mdf["variable"] = beta_float[i]
    conc_mdf.append(mdf)
cdf = pd.concat(conc_mdf)
cdf_02 = pd.concat(conc_02)

print(cdf_02)
cdf['value'] = np.where(cdf['value']==1, "With inhibition", 'Without inhibition')
sns.boxplot(x = "variable", hue = "value", y="size", data=cdf, flierprops={'marker': 'x', 'markersize': 5}, width=.7, palette=['lightskyblue', 'lightcoral'], linewidth=.5, ax=axes[1])
sns.boxplot(x = "Trial", y="size", data=cdf_02, flierprops={'marker': 'x', 'markersize': 5}, width=.5, palette=['lightcoral'], linewidth=.5, ax=axes[0])
axes[1].set_xlabel(r'Synaptic plasticity $(\beta)$')
axes[0].set_xlabel(r'Synaptic plasticity $(\beta)$')
axes[1].set_ylabel(r'Size $(\left|A\right|)$')
axes[0].set_ylabel(r'Size $(\left|A\right|)$')
axes[0].spines[['right', 'top']].set_visible(False)
axes[1].spines[['right', 'top']].set_visible(False)
legend = axes[1].legend(frameon = False)
plt.show()
'''

#7 - Figure 4-c
'''
beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]
f, axes = plt.subplots(1,1)
conc_cdf = []
for i in range(0,5):
    df_emax = filter(pd.read_csv("{0}ormation_02\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0))
    df_emax["variable"] = beta_float[i]
    conc_cdf.append(df_emax)
cdf = pd.concat(conc_cdf)
sns.boxplot(x = "variable", y="density", data=cdf, flierprops={'marker': 'x', 'markersize': 5}, width=.7, palette=['lightskyblue'], linewidth=.5)
axes.legend(frameon=False)
axes.set_xlabel(r'Synaptic plasticity $(\beta)$')
axes.set_ylabel(r"Synaptic density $(D_{A})$")
axes.spines[['right', 'top']].set_visible(False)
plt.show()
'''

#8 - Figure 4-e
'''
norm = mcolors.Normalize(0, 60)

fig,(axes_1,axes_2) = plt.subplots(2,2)
df_A = pd.read_csv("overlap_matrix\matrix_assembly", index_col=0)
df_S = pd.read_csv("overlap_matrix\matrix_stimuli", index_col=0)
print(df_A)
print(df_S)

axes_1[0] = sns.heatmap(data=df_S, fmt='d', cmap='YlOrBr', cbar=True, square=True, ax=axes_1[0],cbar_kws={'label': 'Number of neurons'})
axes_1[0].tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True, length=0)

axes_1[1] = sns.heatmap(data=df_A, fmt='d', cmap='crest', cbar=True, square=True, ax=axes_1[1], norm=norm,cbar_kws={'label': 'Number of neurons'})
axes_1[1].tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True, length=0)


df_Ak = pd.read_csv("overlap_matrix\matrix_assembly_k", index_col=0)
df_Sk = pd.read_csv("overlap_matrix\matrix_stimuli_k", index_col=0)
print(df_Ak)
print(df_Sk)

axes_2[0] = sns.heatmap(data=df_Sk, fmt='d', cmap='YlOrBr', cbar=True, square=True, ax=axes_2[0],cbar_kws={'label': 'Number of neurons'})
axes_2[0].tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True, length=0)

axes_2[1] = sns.heatmap(data=df_Ak, fmt='d', cmap='crest', cbar=True, square=True, ax=axes_2[1], norm=norm, cbar_kws={'label': 'Number of neurons'})
axes_2[1].tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True, length=0)
plt.show()
'''

#9 - Figure 4-f
'''
data_emax_overlap = []
data_k_win_overlap = []
for i in range(0,100):
    df_matrix_E = pd.read_csv("overlap_matrix\matrix_assembly_emax_{0}".format(i), index_col=0)
    mask = np.ones(df_matrix_E.shape, dtype=bool)
    mask[np.triu_indices(len(df_matrix_E))] = False
    new_matrix = df_matrix_E.to_numpy()[mask]
    data_emax_overlap += list(new_matrix)

    df_matrix_K = pd.read_csv("overlap_matrix\matrix_assembly_k_{0}".format(i), index_col=0)
    mask = np.ones(df_matrix_K.shape, dtype=bool)
    mask[np.triu_indices(len(df_matrix_K))] = False
    new_matrix = df_matrix_K.to_numpy()[mask]
    data_k_win_overlap += list(new_matrix)

print(median(data_emax_overlap))
print(median(data_k_win_overlap))

fig,axes = plt.subplots(4,1,sharex=True)
sns.histplot(data=data_emax_overlap, color="lightskyblue" ,ax=axes[1], bins = range(10), label = "Adapted model")
axes[1].legend(frameon=False)
axes[1].set_ylabel(r"#$|A_{i}\cap A_{j}|$")
axes[1].spines[['right', 'top']].set_visible(False)
sns.boxplot(data=data_emax_overlap, ax=axes[0], orient="h", color="lightskyblue",width=.2,flierprops={'marker': 'x', 'markersize': 5})
axes[0].set(yticks=[])
sns.despine(ax=axes[1])
sns.despine(ax=axes[0], left=True)

sns.histplot(data=data_k_win_overlap, color="lightcoral",ax=axes[3], bins = range(10), label = "Original model")
axes[3].legend(frameon=False)
axes[3].set_xlabel(r'$|A_{i}\cap A_{j}|$')
axes[3].set_ylabel(r"#$|A_{i}\cap A_{j}|$")
axes[3].spines[['right', 'top']].set_visible(False)
axes[3].set_xticks(np.arange(0,20,2))
sns.boxplot(data=data_k_win_overlap, ax=axes[2], orient="h", color="lightcoral",width=.2,flierprops={'marker': 'x', 'markersize': 5})
axes[2].set(yticks=[])
sns.despine(ax=axes[3])
sns.despine(ax=axes[2], left=True)
plt.show()
'''

#10 - Figure 5-a
'''
fig, axes = plt.subplots(1,2)
stimu = ["50","100","150","250","500"]
prob = ['1','2','3','4','5']
prob_float = [.1,.2,.3,.4,.5]
df_list = []
suc_list = []
for i in range(0,len(prob)):
    df = pd.DataFrame(pd.read_csv("limit\size_prob_0_"+ prob[i], index_col=0).assign(Trial = prob_float[i]))
    print("PROB: ", prob_float[i])
    print(len(df))
    df_list.append(df)
cdf = pd.concat(df_list)
print(cdf)
sns.boxplot(x = "Trial", y= "0", data=cdf, ax = axes[0], color="lightskyblue",flierprops={'marker': 'x', 'markersize': 5})
axes[0].set_ylabel(r"Size ($|A|$)")
axes[0].set_xlabel(r"Synaptic connection probability($p_{s}$)")
axes[1].set_ylabel(r"Normalized sucess rate ")
axes[1].set_xlabel(r"Synaptic connection probability ($p_{s}$)")
axes[0].spines[['right', 'top']].set_visible(False)
axes[1].spines[['right', 'top']].set_visible(False)
interval = list(np.arange(50,251,50))
interval.append(500)
axes[1].plot([.1,.2,.3,.4,.5],[58/200,107/200,138/200,170/200,179/200],color="lightskyblue")
plt.show()
'''

#11 - Figure 5-b
'''
fig, axes = plt.subplots(1,2)
stimu = ["50","100","150","250","500"]
stimu_float = [50,100,150,250,500]
df_list = []
suc_list = []
for i in range(0,len(stimu)):
    df = pd.DataFrame(pd.read_csv("limit\size_stimuli_"+ stimu[i], index_col=0).assign(Trial = stimu_float[i]))
    print("Stimu: ", stimu_float[i])
    print(len(df))
    df_list.append(df)
cdf = pd.concat(df_list)
print(cdf)
sns.boxplot(x = "Trial", y= "0", data=cdf, ax = axes[0], color="lightskyblue",flierprops={'marker': 'x', 'markersize': 5})
axes[0].set_ylabel(r"Size ($|A|$)")
axes[0].set_xlabel(r"Stimulus size ($k_{s}$)")
axes[1].set_ylabel(r"Normalized sucess rate")
axes[1].set_xlabel(r"Stimulus size ($k_{s}$)")
axes[0].spines[['right', 'top']].set_visible(False)
axes[1].spines[['right', 'top']].set_visible(False)

axes[1].plot([50,100,150,250,500],[66/200,146/200,161/200,189/200,198/200],color="lightskyblue")
plt.show()

'''

#==== STATISTICAL ANALYSIS (Nonparametrical-tests) ====#

#Kruskal + Dunn-Test (T) Table 3
'''
import scikit_posthocs as sp

beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]
f, axes = plt.subplots(1,1)
conc_mdf = []

for i in range(0,5):
    df_emax = filter(pd.read_csv("{0}ormation_02\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0))
    df_kwin = pd.read_csv("{0}ormation_without_emax\{1}ormation_0_{2}_k_winners_".format("f","f",beta[i]), index_col=0)
    print("==== BETA:{0} ====".format(beta_float[i]))
    print("##Adapted_Model")
    print(df_emax.describe())
    print("##Original Model")
    print(df_kwin.describe())
    print()
    df_emax["plasticity"] = beta_float[i]
    conc_mdf.append(df_emax)

cdf = pd.concat(conc_mdf)
#print(cdf)
result = sp.posthoc_dunn(cdf, 'time', 'plasticity', 'bonferroni')
print(result)
#    df_kwin = pd.read_csv("{0}ormation_without_emax\{1}ormation_0_{2}_k_winners_".format("f","f",beta[i]), index_col=0).assign(Trial = 1)
'''

#Mann-Whitney (T) Table 3
'''
import scipy.stats as stats

beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]
for i in range(0,5):
    df_emax = filter(pd.read_csv("{0}ormation_02\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0))
    df_kwin = pd.read_csv("{0}ormation_without_emax\{1}ormation_0_{2}_k_winners_".format("f","f",beta[i]), index_col=0)
    
    stat, p = stats.mannwhitneyu(df_emax['time'],df_kwin['time'], alternative = 'two-sided')
    print("Beta {0}: ".format(beta_float[i]), p)
    print("##Adapted_Model")
    print(df_emax.describe())
    print("##Original_Model")
    print(df_kwin.describe())
    print("=======================")
    print()
'''

#Mann-Whitney (Size, density) Table 4
'''
import scipy.stats as stats
import scikit_posthocs as sp

beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]
cdf_concat = []
for i in range(0,5):
    df_with = filter(pd.read_csv("{0}ormation_02\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0))
    df_without = filter(pd.read_csv("{0}ormation_without\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0))
    df_with['plasticity'] = beta_float[i]
    
    stat, p = stats.mannwhitneyu(df_with['size'],df_without['size'], alternative = 'two-sided')
    print("Beta {0}: ".format(beta_float[i]), p)
    stat, p_with = shapiro(df_with["density"])
    stat, p_without = shapiro(df_without["density"])
    print("Shapiro with: ", p_with)
    print("Shapiro without: ", p_without)
    #print(df_with.describe())
    #print(df_without.describe())
    print(df_with)
    print(df_without)
    print("=======================")
    print()
    cdf_concat.append(df_with)

cdf = pd.concat(cdf_concat)
result_size = sp.posthoc_dunn(cdf, 'size', 'plasticity', 'bonferroni')
result_density = sp.posthoc_dunn(cdf, 'density', 'plasticity', 'bonferroni')
print("Kruskal Size")
print(result_size)
print("Kruskal Density")
print(result_density)
#result = sp.posthoc_dunn(cdf, 'time', 'plasticity', 'bonferroni')
'''

#Mann-Whitney (Overlap)
'''
import scipy.stats as stats
import scikit_posthocs as sp


data_emax_overlap = []
data_k_win_overlap = []
for i in range(0,100):
    df_matrix_E = pd.read_csv("overlap_matrix\matrix_assembly_emax_{0}".format(i), index_col=0)
    mask = np.ones(df_matrix_E.shape, dtype=bool)
    mask[np.triu_indices(len(df_matrix_E))] = False
    new_matrix = df_matrix_E.to_numpy()[mask]
    data_emax_overlap += list(new_matrix)

    df_matrix_K = pd.read_csv("overlap_matrix\matrix_assembly_k_{0}".format(i), index_col=0)
    mask = np.ones(df_matrix_K.shape, dtype=bool)
    mask[np.triu_indices(len(df_matrix_K))] = False
    new_matrix = df_matrix_K.to_numpy()[mask]
    data_k_win_overlap += list(new_matrix)

df_E = pd.DataFrame(data_emax_overlap)
df_K = pd.DataFrame(data_k_win_overlap)

print(df_E.describe())

print(df_K.describe())

stat, p = stats.mannwhitneyu(df_E,df_K, alternative = 'two-sided')
print("p-value:", p)
'''

#Mann-Whitney & Kruskal-Wallis: Recovery + Plot (8 - Figure 4-d / Table 5)
'''
import scipy.stats as stats
import scikit_posthocs as sp

beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]

size_stimuli_emax = [50,100,150,200]
size_stimuli_kwin = [9,18,28,37]
f, axes = plt.subplots(1,1)

conc_mdf = []
conc_emax = []

beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]
conc_mdf = []
conc_emax = []
for i in range(0,5):
    df_e_max = pd.read_csv("recovery\{1}ecovery_e_max_0_{0}_15".format(beta[i],"r"), index_col=0).assign(Trial = 0)
    df_k_win = pd.read_csv("recovery\{1}ecovery_k_win_0_{0}_15".format(beta[i],"r"), index_col=0).assign(Trial = 1)
    print("BETA: {0}".format(beta_float[i]))
    stat, p = stats.mannwhitneyu(df_e_max['0'],df_k_win['0'], alternative = 'two-sided')
    print("Mann-Whitney test: ", p)
    print(df_e_max.describe())
    print(df_k_win.describe())
    print()
    cdf_t = pd.concat([df_e_max,df_k_win])
    mdf = pd.melt(cdf_t, id_vars=["0"], value_vars=["Trial"])
    mdf["variable"] = beta_float[i]
    df_e_max["variable"] = beta_float[i]
    conc_mdf.append(mdf)
    conc_emax.append(df_e_max)
cdf_1 = pd.concat(conc_mdf)
cdf_emax_1 = pd.concat(conc_emax)

result = sp.posthoc_dunn(cdf_emax_1, '0', 'variable', 'bonferroni')
print("Kruskal-Wallis")
print(result)


#cdf['value'] = np.where(cdf['value']==1, 'k-winners-take-all', 'E%-winners-take-all')
cdf_1['value'] = np.where(cdf_1['value']==1, 'Original model', 'Adapted model')
#sns.boxplot(x = "variable", hue = "value", y="0", data=cdf, flierprops={'marker': 'x', 'markersize': 5}, width=.5, palette=['lightskyblue', 'lightcoral'], linewidth=.5, ax=axes[1])
sns.boxplot(x = "variable", hue = "value", y="0", data=cdf_1, flierprops={'marker': 'x', 'markersize': 5}, width=.5, palette=['lightskyblue','lightcoral'], linewidth=.5, ax=axes)
#axes[1].set_xlabel(r'Tamanho do estímulo $(\%)$')
axes.set_xlabel(r'Synaptic plasticity $(\beta)$')
axes.spines[['right', 'top']].set_visible(False)
#axes[1].set_ylabel(r'Porção recuperada $(\%)$')
axes.set_ylabel(r'Recovered portion ($|A|_{rec}$/$|A|$)')
#legend = axes[1].legend(frameon = False)
legend = axes.legend(frameon = False)
plt.show()
'''
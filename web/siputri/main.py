from flask import Flask, render_template
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import t

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

df = pd.read_excel('final.xlsx')
k = 4
n = len(df)- k
df["total_x1"] = df.iloc[:, 6:14].sum(axis=1)
df["total_x2"] = df.iloc[:, 14:19].sum(axis=1)
df["total_x3"] = df.iloc[:, 19:22].sum(axis=1)
df["total_x4"] = df.iloc[:, 22:24].sum(axis=1)
df["total_y"] = df.iloc[:, 24:26].sum(axis=1)
Xvar = df[['total_x1','total_x2','total_x3','total_x4']]
Xvar = sm.add_constant(Xvar)
Yvar = df['total_y']
olsmod = sm.OLS(Yvar,Xvar).fit()
df['Ypredict'] = round(olsmod.predict(Xvar),4)
df['Residual'] = round(olsmod.resid,4)

def uji_normalitas_residual():
    n_stats, p_val = sm.stats.diagnostic.kstest_normal(
    df['Residual'],dist='norm', pvalmethod='table')
    print('P_value Kolmogorov-Smirnov :', round(p_val,3))

    if p_val > 0.05:
        return ("Data terdistribusi normal")
    elif p_val < 0.05:
        return ("Data terdistribusi tidak normal")
    
def uji_multikol():
    vif_data = pd.DataFrame()
    vif_data["variabel"] = Xvar.columns
    vif_data["VIF"] = [variance_inflation_factor(Xvar.values,i)for i in range(len(Xvar.columns))]
    vif_data['Kesimpulan'] = ['Terjadi multikol' if vif > 10 else 'Tidak terjadi multikol' for vif in vif_data["VIF"]]
    vif_data = vif_data.drop(vif_data[vif_data['variabel'] == 'const'].index)
    return(vif_data)

def uji_hetero():
    return sns.residplot(x=df["Ypredict"], y=df["Residual"]).set(title='Predict vs Residual')

def auto_korelasi():
    durbinWatson = durbin_watson(df["Residual"])
    print("Nilai Auto Korelasi :", durbinWatson)
    tb_dw = pd.read_excel('tabel_dw.xlsx')
    n = len(df)
    dl_du = tb_dw[tb_dw['n'] == n]
    dl_value = dl_du['dl'].values[0]
    du_value = dl_du['du'].values[0]
    print(f"N = {n} maka DL = {dl_value} dan DU = {du_value}")

    if durbinWatson >= du_value and durbinWatson < 4 - du_value:
        return("Tidak terjadi Auto Korelasi")
    elif durbinWatson < dl_value and durbinWatson > 4 - dl_value:
        return("Terjadi auto korelasi")

def uji_t():
    df_t = pd.DataFrame(columns=['dk', 'α=0.05'])
    df_t['dk'] = [i for i in range(1, len(df))]
    df_t['α=0.05'] = [round(t.ppf(0.975, df), 3) for df in df_t['dk']]
    nilai_t = df_t[df_t['dk'] == n]['α=0.05'].values[0]
    # print("Nilai t tabel dari", n ,"adalah", nilai_t)
    total_x1 = olsmod.tvalues['total_x1']
    total_x2 = olsmod.tvalues['total_x2']
    total_x3 = olsmod.tvalues['total_x3']
    total_x4 = olsmod.tvalues['total_x4']
    const = olsmod.tvalues['const']
    if total_x1 > nilai_t:
        print("Nilai total_x1 berpengaruh pada terhadap Y.")
    else:
        print("Nilai total_x1 tidak berpengaruh pada terhadap Y.")
        
    if total_x2 > nilai_t:
        print("Nilai total_x2 berpengaruh pada terhadap Y.")
    else:
        print("Nilai total_x2 tidak berpengaruh pada terhadap Y.")
        
    if total_x3 > nilai_t:
        print("Nilai total_x3 berpengaruh pada terhadap Y.")
    else:
        print("Nilai total_x3 tidak berpengaruh pada terhadap Y.")
        
    if total_x4 > nilai_t:
        print("Nilai total_x4 berpengaruh pada terhadap Y.")
    else:
        print("Nilai total_x4 tidak berpengaruh pada terhadap Y.")

def uji_regresi():
    # Nilai Linear Regression
    bebas = ['total_x1','total_x2','total_x3','total_x4']
    X = df[bebas]
    y = df['total_y']

    model = LinearRegression().fit(X,y)

    koefisien = model.coef_
    intercept = model.intercept_

    print("Intercept : ", intercept)
    list(zip(bebas, koefisien))

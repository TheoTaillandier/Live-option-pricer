import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import requests
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from urllib.parse import unquote
import json
from bs4 import BeautifulSoup
from scipy.interpolate import make_interp_spline

def parse_cbot_price(price_str):
    "Convertit en float les prix avec tiret"
    if isinstance(price_str, str) and '-' in price_str:
        base, frac = price_str.split('-')
        try:
            return float(base) + float(frac)/8
        except Exception:
            return float(base)
    else:
        try:
            return float(price_str)
        except Exception:
            return None

def get_barchart_skew(ticker):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...")
    driver = webdriver.Chrome(options=options)
    url = f"https://www.barchart.com/futures/quotes/{ticker}/volatility-greeks?futuresOptionsView=merged"
    driver.get(url)
    driver.implicitly_wait(7)
    html = driver.page_source
    cookies_driver = driver.get_cookies()
    driver.quit()

    #Spot extraction
    match =re.search(r'"lastPrice":"([\d\.]+)s?"', html)
    if not match:
        match =re.search(r'"lastPrice":"([\d\.]+)"', html)
    if not match:
        match =re.search(r'"lastPrice":"([\d\.]+)', html)
    if match:
        spot = parse_cbot_price(match.group(1))
    else:
        raise ValueError("Spot non trouvé dans le code HTML")

    cookies = {cook['name']: cook['value'] for cook in cookies_driver}
    headers = {
        'accept': 'application/json',
        'referer': url,
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
        'x-xsrf-token': unquote(cookies['XSRF-TOKEN']),
    }
    params = {
        'symbol': ticker,
        'list': 'futures.options',
        'fields': 'strikePrice,optionType,baseSymbol,lastPrice,optImpliedVolatility,delta,gamma,theta,vega,impliedVolatilitySkew,lastPrice,tradeTime,longSymbol,symbolCode,symbolType',
        'meta': 'field.shortName,field.description,field.type,lists.lastUpdate',
        'groupBy': 'optionType',
        'orderBy': 'strikePrice',
        'orderDir': 'asc',
        'raw': '1',
    }
    response = requests.get(
        'https://www.barchart.com/proxies/core-api/v1/quotes/get',
        params=params,
        cookies=cookies,
        headers=headers,
    )
    rep = json.loads(response.content)
    data_call = rep['data']['Call']
    data_put = rep['data']['Put']

    df_call = pd.DataFrame(data_call)
    df_call['optionType'] = 'Call'
    df_put = pd.DataFrame(data_put)
    df_put['optionType'] = 'Put'
    df_all = pd.concat([df_call, df_put], ignore_index=True)
    df_all = df_all[['strikePrice', 'optImpliedVolatility', 'optionType']].dropna()
    df_all['strikePrice'] = df_all['strikePrice'].apply(parse_cbot_price)
    df_all['optImpliedVolatility'] = df_all['optImpliedVolatility'].str.replace('%', '', regex=False)
    df_all['optImpliedVolatility'] = pd.to_numeric(df_all['optImpliedVolatility'], errors='coerce')
    df_all.dropna(inplace=True)
    df_all = df_all[df_all['optImpliedVolatility'] > 0]

    puts_otm = df_all[(df_all['optionType'] == 'Put') & (df_all['strikePrice'] < spot)]
    calls_otm = df_all[(df_all['optionType'] == 'Call') & (df_all['strikePrice'] > spot)]
    df_otm = pd.concat([puts_otm, calls_otm], ignore_index=True)
    df_otm = df_otm.sort_values(by='strikePrice')
    return spot, df_otm

def get_vol_interp_function(df_otm):
    strikes=df_otm['strikePrice'].values
    vols=df_otm['optImpliedVolatility'].values/100
    return interp1d(strikes, vols, kind='cubic', fill_value='extrapolate')

def get_r():
    url = "https://fr.investing.com/rates-bonds/france-10-year-bond-yield"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    if r.status_code==200:
        soup = BeautifulSoup(r.text, "html.parser")
        texte = soup.get_text()
        match = re.search(r"dernier prix:\s*([\d,]+)", texte)
        if match:
            r = float(match.group(1).replace(",", "."))/100
            return r
        else:
            return 0.035
    else:
        return 0.035
    
def get_r_us():
    url = "https://fr.investing.com/rates-bonds/u.s.-10-year-bond-yield"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        soup = BeautifulSoup(r.text, "html.parser")
        texte = soup.get_text()
        match = re.search(r"Dernier prix:\s*([\d,]+)", texte, re.IGNORECASE)
        if not match:
            # Essaye autre format
            match = re.search(r"(\d+,\d+)\s*%", texte)
        if match:
            r = float(match.group(1).replace(",", ".")) / 100
            return r
        else:
            return 0.043  # Valeur par défaut si parsing échoue
    else:
        return 0.043

def get_r_ca():
    url = "https://fr.investing.com/rates-bonds/canada-10-year-bond-yield"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        soup = BeautifulSoup(r.text, "html.parser")
        texte = soup.get_text()
        match = re.search(r"Dernier prix:\s*([\d,]+)", texte, re.IGNORECASE)
        if not match:
            # Essaye autre format
            match = re.search(r"(\d+,\d+)\s*%", texte)
        if match:
            r = float(match.group(1).replace(",", ".")) / 100
            return r
        else:
            return 0.035  # Valeur par défaut si parsing échoue
    else:
        return 0.035


def pricer_option_binomial(S0,K,T,r,sigma,n,option_type="call",position="long"):
    dt=T/n
    u=np.exp(sigma*np.sqrt(dt))
    d=1/u
    p=(np.exp(r*dt)-d)/(u-d)
    stock_tree=np.zeros((n+1,n+1))
    for i in range(n+1):
        for j in range(n+1):
            stock_tree[j,i]=S0*(u**(i-j))*(d**j)
    option_tree=np.zeros((n+1,n+1))
    if option_type== "call":
        option_tree[:, n]= np.maximum(stock_tree[:,n]-K, 0)
    else:
        option_tree[:,n]=np.maximum(K-stock_tree[:,n], 0)
    for i in range(n-1,-1,-1):
        for j in range(i+1):
            early_exercise=max(stock_tree[j,i]-K,0) if option_type== "call" else max(K-stock_tree[j,i], 0)
            continuation_value=np.exp(-r*dt)*(p*option_tree[j,i+1]+(1-p)*option_tree[j+1,i+1])
            option_tree[j,i]=max(early_exercise, continuation_value)
    return option_tree,stock_tree

def calcul_delta(S0,K,T,r,sigma,n,option_type,position="long"):
    option_tree,stock_tree=pricer_option_binomial(S0,K,T,r,sigma,n,option_type,position)
    S_up= stock_tree[0,1]
    S_down= stock_tree[1,1]
    V_up= option_tree[0,1]
    V_down= option_tree[1,1]
    delta= (V_up-V_down)/(S_up-S_down)
    return delta if position=="long" else -delta

def calcul_gamma(S0,K,T,r,sigma,n,option_type,position="long",deriv=0.001):
    delta_up= calcul_delta(S0+deriv,K,T,r,sigma,n,option_type)
    delta_down= calcul_delta(S0-deriv,K,T,r,sigma,n,option_type)
    S_up= S0+deriv
    S_down= S0-deriv
    gamma= (delta_up-delta_down)/(S_up-S_down)*100
    return gamma if position== "long" else -gamma

def calcul_vega(S0,K,T,r,sigma,n,option_type,position="long",deriv=0.001):
    price_up= pricer_option_binomial(S0,K,T,r,sigma+deriv,n,option_type)[0]
    price_down= pricer_option_binomial(S0,K,T,r,sigma-deriv,n,option_type)[0]
    vega= (price_up[0,0]-price_down[0,0])/(2*deriv)/2
    return vega if position== "long" else -vega

# --- Interface graphique ---
ASSET_CATEGORIES = {
    "GRAINS & Oilseeds": [
        ("WHEAT (CBOT)", "ZW"),
        ("Milling Wheat (Euronext)", "ML"),
        ("CORN (CBOT)", "ZC"),
        ("CORN (Euronext)", "XB"),
        ("SOYBEAN (CBOT)", "ZS"),
        ("SoyBEAN Meal (CBOT)", "ZM"),
        ("SoyBEAN Oil (CBOT)", "ZL"),
        ("OAT (CBOT)", "ZO"),
        ("Rough Rice (CBOT)", "ZR"),
        ("Hars Red Winter Wheat (CBOT)", "KE"),
        ("Spring Wheat (MIAX)", "MW"),
    ],
    "Energies": [
        ("Crude Oil WTI (NYMEX)", "CL"),
        ("Crude Oil WTI (ICE/EU)", "WI"),
        ("Crude Oil Brent (NYMEX)", "QA"),
        ("Crude Oil Brent (ICE/EU)", "CB"),
        ("Gasoline RBOB (NYMEX)", "RB"),
        ("Natural Gas (NYMEX)", "NG"),
        ("Ethanol (NYMEX)", "FL"),
        ("ULSD NY Harbor (NYMEX)", "HO"),
    ],
    "Metals": [
        ("Gold (COMEX)", "GC"),
        ("Silver (COMEX)", "SI"),
        ("Platinum (NYMEX)", "PL"),
        ("Aluminum (COMEX)", "AL"),
        ("Palladium (NYMEX)", "PA"),
        ("High Grade Copper (COMEX)", "HG"),
    ],
    "Soft": [
        ("Cotton (ICE/US)", "CT"),
        ("Orange Juice (ICE/US)", "OJ"),
        ("Coffee (ICE/US)", "KC"),
        ("Coffee Robusta 10T (ICE/EU)", "RM"),
        ("Cocoa (ICE/US)", "CC"),
        ("Cocoa (ICE/EU)", "CA"),
        ("Sugar (ICE/US)", "SB"),
        ("White Sugar (ICE/EU)", "SW"),
        ("Lumber (CME)", "LB"),
        ("CANOLA (ICE/CA)", "RS"),
    ]
}
MONTH_CODES = [
    ("Janvier", "F"),
    ("Février", "G"),
    ("Mars", "H"),
    ("Avril", "J"),
    ("Mai", "K"),
    ("Juin", "M"),
    ("Juillet", "N"),
    ("Août", "Q"),
    ("Septembre", "U"),
    ("Octobre", "V"),
    ("Novembre", "X"),
    ("Décembre", "Z"),
]
YEARS = [str(y) for y in range(2024, 2032)]


class OptionPricerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Option Pricer - Espace Interactif")
        self.geometry("900x700")
        self.spot = None
        self.df_otm = None
        self.vol_interp = None
        self.r = None
        self.nb_legs = 1
        self.legs_entries = []
        self.create_ticker_frame()

    def clear_frame(self):
        for widget in self.winfo_children():
            widget.destroy()

    def create_ticker_frame(self):
        self.clear_frame()
        tk.Label(self, text="Sélectionnez la catégorie, l'actif, le mois et l'année :", font=('Arial', 14)).pack(pady=15)

        # Catégorie
        self.category_var = tk.StringVar()
        categories = list(ASSET_CATEGORIES.keys())
        self.category_combo = ttk.Combobox(self, textvariable=self.category_var, values=categories, state="readonly", width=25)
        self.category_combo.pack(pady=3)
        self.category_combo.bind("<<ComboboxSelected>>", self.update_asset_combo)
    
        # Actif (sera rempli dynamiquement)
        self.asset_var = tk.StringVar()
        self.asset_combo = ttk.Combobox(self, textvariable=self.asset_var, values=[], state="readonly", width=35)
        self.asset_combo.pack(pady=3)
    
        # Mois
        self.month_var = tk.StringVar()
        month_names = [m[0] for m in MONTH_CODES]
        ttk.Combobox(self, textvariable=self.month_var, values=month_names, state="readonly", width=15).pack(pady=3)
    
        # Année
        self.year_var = tk.StringVar()
        ttk.Combobox(self, textvariable=self.year_var, values=YEARS, state="readonly", width=8).pack(pady=3)
    
        tk.Button(self, text="Charger Skew & Spot", command=self.load_skew_spot_thread).pack(pady=15)
        self.status_label = tk.Label(self, text="", font=('Arial', 12), fg="blue")
        self.status_label.pack(pady=10)

    def update_asset_combo(self, event=None):
        category = self.category_var.get()
        assets = ASSET_CATEGORIES.get(category, [])
        asset_names = [a[0] for a in assets]
        self.asset_combo['values'] = asset_names
        self.asset_var.set('')


    def load_skew_spot_thread(self):
        threading.Thread(target=self.load_skew_spot_from_combo).start()

    def load_skew_spot_from_combo(self):
        category = self.category_var.get()
        asset_name = self.asset_var.get()
        month_name = self.month_var.get()
        year = self.year_var.get()
        if not (category and asset_name and month_name and year):
            messagebox.showerror("Erreur", "Veuillez sélectionner la catégorie, l'actif, le mois et l'année.")
            return
        asset_code = dict(ASSET_CATEGORIES[category])[asset_name]
        month_code = dict(MONTH_CODES)[month_name]
        year_code = year[-1]
        ticker = f"{asset_code}{month_code}{year_code}".upper()
        try:
            self.status_label.config(text="Récupération du skew de volatilité...")
            self.update_idletasks()
            self.spot, self.df_otm = get_barchart_skew(ticker)
            self.status_label.config(text="Interpolation de la volatilité en fonction du strike...")
            self.update_idletasks()
            self.vol_interp = get_vol_interp_function(self.df_otm)
            self.status_label.config(text="Récupération du taux sans risque...")
            self.update_idletasks()
            if "(CBOT)" in asset_name or "(ICE/US)" in asset_name or "(NYMEX)" in asset_name or "(CME)" in asset_name or "(MIAX)" in asset_name:
                self.r = get_r_us()
            elif "(ICE/CA)" in asset_name:
                self.r=get_r_ca()
            else:
                self.r = get_r()
            self.show_skew_plot()
        except Exception as e:
            self.status_label.config(text="")
            messagebox.showerror("Erreur", f"Impossible de récupérer les données : {e}")


    def load_skew_spot(self):
        ticker = self.ticker_entry.get().strip().upper()
        if not ticker:
            messagebox.showerror("Erreur", "Veuillez saisir un ticker valide.")
            return
        try:
       # Étape 1 : récupération du skew
           self.status_label.config(text="Récupération du skew de volatilité...")
           self.update_idletasks()
           self.spot, self.df_otm = get_barchart_skew(ticker)

       # Étape 2 : interpolation
           self.status_label.config(text="Interpolation de la volatilité en fonction du strike...")
           self.update_idletasks()
           self.vol_interp = get_vol_interp_function(self.df_otm)

       # Étape 3 : taux sans risque
           self.status_label.config(text="Récupération du taux sans risque...")
           self.update_idletasks()
           self.r = get_r()

       # Affichage du skew
           self.show_skew_plot()

        except Exception as e:
           self.status_label.config(text="")
           messagebox.showerror("Erreur", f"Impossible de récupérer les données : {e}")

    def show_skew_plot(self):
        self.clear_frame()
        tk.Label(self, text=f"Spot détecté : {self.spot:.2f}", font=('Arial', 14)).pack(pady=5)
        fig, ax = plt.subplots(figsize=(7, 4))
        x = self.df_otm['strikePrice'].values
        y = self.df_otm['optImpliedVolatility'].values
        if len(x) >= 4:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spline = make_interp_spline(x, y, k=3)
            y_smooth = spline(x_smooth)
            ax.plot(x_smooth, y_smooth, label="Volatility Skew (OTM)", color='blue')
        ax.plot(x, y, 'o', label='Volatilité implicite (%)', color='orange')
        ax.axvline(self.spot, color='red', linestyle='--', label=f'Spot = {self.spot}')
        ax.set_xlabel('Strike')
        ax.set_ylabel('Volatilité implicite (%)')
        ax.set_title('Skew de volatilité')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().pack()
        canvas.draw()
        tk.Label(self, text=f"Taux sans risque utilisé : {self.r:.3%}", font=('Arial', 12)).pack(pady=5)
        tk.Label(self, text="Combien de legs voulez-vous pricer ? (1, 2 ou 3)", font=('Arial', 12)).pack(pady=10)
        self.nb_legs_var = tk.IntVar(value=1)
        ttk.Combobox(self, textvariable=self.nb_legs_var, values=[1,2,3], state="readonly", width=5).pack()
        tk.Button(self, text="Continuer", command=self.create_legs_frame).pack(pady=15)

    def create_legs_frame(self):
        self.nb_legs = self.nb_legs_var.get()
        self.clear_frame()
        self.legs_entries = []
        for i in range(self.nb_legs):
            frame = tk.LabelFrame(self, text=f"Leg {i+1}", padx=10, pady=10)
            frame.pack(padx=10, pady=10, fill="x")
            tk.Label(frame, text=f"Strike :", width=14).grid(row=0, column=0, sticky="e")
            strike_entry = tk.Entry(frame)
            strike_entry.grid(row=0, column=1)
            tk.Label(frame, text="Maturité (jours) :", width=14).grid(row=1, column=0, sticky="e")
            maturity_entry = tk.Entry(frame)
            maturity_entry.grid(row=1, column=1)
            tk.Label(frame, text="Type d'option :", width=14).grid(row=2, column=0, sticky="e")
            type_var = tk.StringVar(value="call")
            ttk.Combobox(frame, textvariable=type_var, values=["call", "put"], state="readonly", width=8).grid(row=2, column=1)
            tk.Label(frame, text="Position :", width=14).grid(row=3, column=0, sticky="e")
            pos_var = tk.StringVar(value="long")
            ttk.Combobox(frame, textvariable=pos_var, values=["long", "short"], state="readonly", width=8).grid(row=3, column=1)
            self.legs_entries.append({
                "strike": strike_entry,
                "maturity": maturity_entry,
                "type": type_var,
                "pos": pos_var
            })
        tk.Button(self, text="Calculer le pricing", command=self.price_strategy_thread).pack(pady=20)
        self.result_text = tk.Text(self, height=12, font=('Consolas', 12))
        self.result_text.pack(fill="both", expand=True, padx=10, pady=10)

    def price_strategy_thread(self):
        threading.Thread(target=self.price_strategy).start()

    def price_strategy(self):
        try:
            legs = []
            for i, leg in enumerate(self.legs_entries):
                K = float(leg["strike"].get())
                T = float(leg["maturity"].get()) / 365
                option_type = leg["type"].get()
                position = leg["pos"].get()
                sigma = float(self.vol_interp(K))
                n = 500
                option_tree, _ = pricer_option_binomial(self.spot, K, T, self.r, sigma, n, option_type, position)
                price = option_tree[0, 0]
                delta = calcul_delta(self.spot, K, T, self.r, sigma, n, option_type, position)
                gamma = calcul_gamma(self.spot, K, T, self.r, sigma, n, option_type, position)
                vega = calcul_vega(self.spot, K, T, self.r, sigma, n, option_type, position)
                signe = 1 if position == "long" else -1
                legs.append({
                    "prix": price * signe,
                    "delta": delta,
                    "gamma": gamma,
                    "vega": vega,
                    "sigma": sigma
                })
            total_price = sum(leg["prix"] for leg in legs)
            total_delta = sum(leg["delta"] for leg in legs)
            total_gamma = sum(leg["gamma"] for leg in legs)
            total_vega = sum(leg["vega"] for leg in legs)
            res = ""
            for i, leg in enumerate(legs):
                res += f"--- Résultats pour la leg {i+1} (vol : {leg['sigma']:.2%}) ---\n"
                res += f"Prix: {leg['prix']:.4f}\nDelta: {leg['delta']*100:.3f}%\nGamma: {leg['gamma']:.4f}\nVega: {leg['vega']:.4f}\n\n"
            res += "--- Résultat global de la stratégie ---\n"
            res += f"Prix total: {total_price:.4f}\nDelta total: {total_delta*100:.3f}%\nGamma total: {total_gamma:.4f}\nVega total: {total_vega:.4f}\n"
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, res)
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur dans le pricing : {e}")

if __name__ == "__main__":
    app = OptionPricerApp()
    app.mainloop()
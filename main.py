import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

H = 5                     
POS_RATE_FLOOR = 0.10     
TEST_DAYS = 300          

df = yf.download("^GSPC", start="1990-01-01", progress=False, actions=False, auto_adjust=False)
if df.empty:
    df = yf.download("SPY", start="1990-01-01", progress=False, actions=False, auto_adjust=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]
df = df[["Open","High","Low","Close","Volume"]].dropna()

# target: H-day ahead up/down
df["Future"] = df["Close"].shift(-H)
df["Target"] = (df["Future"] > df["Close"]).astype(int)

# features
df["ret1"] = df["Close"].pct_change(1)
df["ret5"] = df["Close"].pct_change(5)
df["ret20"] = df["Close"].pct_change(20)
df["ma5_rel"] = df["Close"].rolling(5).mean()/df["Close"] - 1
df["ma20_rel"] = df["Close"].rolling(20).mean()/df["Close"] - 1
df["std20"] = df["Close"].pct_change().rolling(20).std()
df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]
df["vol_logdiff5"] = np.log1p(df["Volume"]).diff(5)

# RSI(14)
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
rs = gain / loss
df["rsi14"] = 100 - (100/(1+rs))

# MACD(12,26) + signal(9)
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["macd"] = ema12 - ema26
df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()

# ATR(14) normalized
tr = np.maximum(df["High"]-df["Low"],
                np.maximum((df["High"]-df["Close"].shift()).abs(),
                           (df["Low"]-df["Close"].shift()).abs()))
df["atr14"] = (tr.rolling(14).mean()/df["Close"])

features = [
    "ret1","ret5","ret20","ma5_rel","ma20_rel","std20",
    "hl_spread","vol_logdiff5","rsi14","macd","macd_sig","atr14"
]

# clean
df[features] = df[features].replace([np.inf,-np.inf], np.nan)
df = df.dropna(subset=features+["Target"]).copy()
df[features] = df[features].fillna(0).clip(-5,5)

# split (last TEST_DAYS for test)
train = df.iloc[:-TEST_DAYS].copy()
test  = df.iloc[-TEST_DAYS:].copy()
Xtr, ytr = train[features].values, train["Target"].values
Xte, yte = test[features].values, test["Target"].values


base = RandomForestClassifier(
    n_estimators=600, max_depth=10, min_samples_split=10,
    class_weight="balanced", random_state=42, n_jobs=-1
)
model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
model.fit(Xtr, ytr)


proba_tr = model.predict_proba(Xtr)[:,1]
qs = np.linspace(0.2, 0.8, 61)
cands = np.unique(np.quantile(proba_tr, qs))
best_t, best_prec, best_acc = 0.5, 0.0, 0.0
n = len(proba_tr)
for t in cands:
    pred = (proba_tr >= t).astype(int)
    pos_rate = pred.mean()
    prec = precision_score(ytr, pred, zero_division=0)
    acc = accuracy_score(ytr, pred)
    if pos_rate >= POS_RATE_FLOOR and (prec > best_prec or (prec==best_prec and acc>best_acc)):
        best_t, best_prec, best_acc = float(t), float(prec), float(acc)

proba_te = model.predict_proba(Xte)[:,1]
preds = (proba_te >= best_t).astype(int)

print({"threshold": round(best_t,3), "train_precision_at_t": round(best_prec,3), "train_accuracy_at_t": round(best_acc,3)})
print("accuracy", round(accuracy_score(yte, preds), 4))
print("precision", round(precision_score(yte, preds, zero_division=0), 4))
print("recall", round(recall_score(yte, preds, zero_division=0), 4))
print(confusion_matrix(yte, preds))
print(classification_report(yte, preds, zero_division=0))

pd.DataFrame({"Target": yte, "Pred": preds}, index=test.index).plot()
plt.title("Target vs Pred (H=5 days)")
plt.show()

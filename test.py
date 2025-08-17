import sys
import yfinance as yf
import pandas as pd

print("Using interpreter:", sys.executable)

df = yf.download("AAPL", period="1mo")
print(df.head())

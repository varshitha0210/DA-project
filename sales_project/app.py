from flask import Flask, render_template, request, redirect
import pandas as pd
import os
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

FILE = "data.csv"

if not os.path.exists(FILE):
    df = pd.DataFrame(columns=["Date","Item","Price","Quantity","Day","Weekend"])
    df.to_csv(FILE, index=False)


@app.route('/')
def home():
    df = pd.read_csv(FILE)
    return render_template("index.html",
                           tables=[df.to_html(classes='table table-dark table-striped', index=False)])


# 🔹 ADD DATA
@app.route('/add', methods=['POST'])
def add():
    data = {
        "Date": request.form['date'],
        "Item": request.form['item'],
        "Price": float(request.form['price']),
        "Quantity": float(request.form['quantity']),
        "Day": request.form['day'],
        "Weekend": request.form['weekend']
    }

    df = pd.read_csv(FILE)
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(FILE, index=False)

    return redirect('/')


# 🔹 NORMALIZATION
@app.route('/normalize')
def normalize():
    df = pd.read_csv(FILE)

    df['Price'] = (df['Price'] - df['Price'].min()) / (df['Price'].max() - df['Price'].min())
    df['Quantity'] = (df['Quantity'] - df['Quantity'].min()) / (df['Quantity'].max() - df['Quantity'].min())

    df.to_csv(FILE, index=False)
    return redirect('/')


# 🔹 ANOMALY
@app.route('/anomaly')
def anomaly():
    df = pd.read_csv(FILE)

    mean = df['Quantity'].mean()
    std = df['Quantity'].std()

    df['Z_score'] = (df['Quantity'] - mean) / std
    df['Anomaly'] = df['Z_score'].abs() > 2

    df.to_csv(FILE, index=False)
    return redirect('/')


# 🔹 ELASTICITY
@app.route('/elasticity')
def elasticity():
    df = pd.read_csv(FILE)

    df['Price_change'] = df['Price'].pct_change()
    df['Qty_change'] = df['Quantity'].pct_change()
    df['Elasticity'] = df['Qty_change'] / df['Price_change']

    df.to_csv(FILE, index=False)
    return redirect('/')


# 🔹 MOVING AVG
@app.route('/moving')
def moving():
    df = pd.read_csv(FILE)
    df['Moving_Avg'] = df['Quantity'].rolling(3).mean()
    df.to_csv(FILE, index=False)
    return redirect('/')


# 🔹 EXPONENTIAL SMOOTHING
@app.route('/forecast')
def forecast():
    df = pd.read_csv(FILE)

    alpha = 0.5
    forecast = [df['Quantity'][0]]

    for i in range(1, len(df)):
        val = alpha * df['Quantity'][i-1] + (1-alpha) * forecast[i-1]
        forecast.append(val)

    df['Forecast'] = forecast
    df.to_csv(FILE, index=False)

    return redirect('/')


# 🔥 APRIORI
@app.route('/apriori')
def run_apriori():
    df = pd.read_csv(FILE)

    basket = pd.crosstab(df['Date'], df['Item'])
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    freq = apriori(basket, min_support=0.2, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.5)

    rules.to_csv("rules.csv", index=False)

    return redirect('/')


# 🔥 PREFIXSPAN (Simplified)
@app.route('/prefixspan')
def prefixspan():
    df = pd.read_csv(FILE)

    sequences = df.groupby('Date')['Item'].apply(list)

    patterns = {}

    for seq in sequences:
        for i in range(len(seq)-1):
            pair = (seq[i], seq[i+1])
            patterns[pair] = patterns.get(pair, 0) + 1

    result = pd.DataFrame(list(patterns.items()), columns=["Sequence", "Count"])
    result.to_csv("sequences.csv", index=False)

    return redirect('/')


# 🔹 PLOT
@app.route('/plot')
def plot():
    df = pd.read_csv(FILE)

    plt.figure()
    plt.plot(df['Quantity'], label='Actual')
    if 'Forecast' in df.columns:
        plt.plot(df['Forecast'], label='Forecast')

    plt.legend()
    plt.title("Sales Trend")
    plt.savefig("static/plot.png")
    plt.close()

    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)
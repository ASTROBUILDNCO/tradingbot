import krakenex
import time
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import requests
import logging
import os
from datetime import datetime, timedelta
import json
import threading
from typing import Dict, List, Optional, Tuple
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator

# Configure logging
logging.basicConfig(filename='kraken_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedAIPoweredKrakenBot:
    def __init__(self, api_key=None, api_secret=None, pair="SOL/USD", trade_amount=3, risk_per_trade=0.01):
        self.api = krakenex.API()
        self.api.key = api_key
        self.api.secret = api_secret
        self.pair = pair
        self.trade_amount = trade_amount
        self.risk_per_trade = risk_per_trade
        self.scaler = MinMaxScaler()
        self.model = self.load_or_build_model()
        self.account_balance = self.fetch_account_balance()

        # UI setup
        self.root = tk.Tk()
        self.root.title(f"Advanced Kraken Trading Bot - {pair}")
        self.root.geometry("1200x800")
        self.create_enhanced_ui()

    def load_or_build_model(self):
        try:
            if os.path.exists("ai_trading_model.h5"):
                from tensorflow.keras.models import load_model
                model = load_model("ai_trading_model.h5")
                logging.info("Loaded pre-trained model.")
            else:
                model = Sequential([
                    LSTM(128, input_shape=(60, 5), return_sequences=True),
                    Dropout(0.2),
                    LSTM(64, return_sequences=False),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dense(4, activation='softmax')
                ])
                model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
                logging.info("Built new AI model.")
            return model
        except Exception as e:
            logging.error(f"Error loading or building AI model: {e}")
            raise

    def fetch_account_balance(self):
        try:
            response = self.api.query_private('Balance')
            if response.get('error'):
                logging.error(f"Error fetching account balance: {response['error']}")
                return 10000
            return float(response['result'].get('ZUSD', 10000))
        except Exception as e:
            logging.error(f"Exception fetching account balance: {e}")
            return 10000

    def fetch_ohlc_data(self, interval=1):
        try:
            response = self.api.query_public('OHLC', {'pair': self.pair, 'interval': interval})
            if response.get('error'):
                logging.error(f"Error fetching OHLC data: {response['error']}")
                return None

            data = response['result'][self.pair]
            df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
            df['close'] = df['close'].astype(float)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        except Exception as e:
            logging.error(f"Error fetching OHLC data: {e}")
            return None

    def classify_market(self, df):
        try:
            sma = SMAIndicator(df['close'], window=20).sma_indicator().iloc[-1]
            rsi = RSIIndicator(df['close'], window=14).rsi().iloc[-1]
            bollinger = BollingerBands(df['close'], window=20).bollinger_wband().iloc[-1]

            features = self.scaler.fit_transform(
                np.array([[df['close'].iloc[-1], sma, rsi, bollinger]])
            )

            prediction = self.model.predict(features.reshape(1, 1, -1))
            market_condition = np.argmax(prediction)

            conditions = ['Bull', 'Bear', 'Sideways', 'Volatile']
            return conditions[market_condition]
        except Exception as e:
            logging.error(f"Error classifying market: {e}")
            return "Unknown"

    def execute_trade(self, action):
        try:
            order = {
                'pair': self.pair,
                'type': action,
                'ordertype': 'market',
                'volume': str(self.trade_amount)
            }

            response = self.api.query_private('AddOrder', order)
            if response.get('error'):
                logging.error(f"Trade error: {response['error']}")
                return

            self.update_ui_log(f"Trade successful: {response['result']}")
        except Exception as e:
            logging.error(f"Error executing trade: {e}")

    def create_enhanced_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Dashboard tab
        dashboard_tab = ttk.Frame(notebook)
        notebook.add(dashboard_tab, text="Dashboard")

        # Trade History tab
        trade_history_tab = ttk.Frame(notebook)
        notebook.add(trade_history_tab, text="Trade History")

        # Settings tab
        settings_tab = ttk.Frame(notebook)
        notebook.add(settings_tab, text="Settings")

        # Dashboard content
        dashboard_frame = ttk.LabelFrame(dashboard_tab, text="Market Overview")
        dashboard_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.market_label = ttk.Label(dashboard_frame, text="Market Condition: Unknown", font=("Helvetica", 16))
        self.market_label.pack(pady=10)

        self.balance_label = ttk.Label(dashboard_frame, text=f"Balance: ${self.account_balance:.2f}", font=("Helvetica", 12))
        self.balance_label.pack(pady=10)

        chart_frame = ttk.LabelFrame(dashboard_tab, text="Price Chart")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.chart_canvas = FigureCanvasTkAgg(self.figure, chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Trade History content
        trade_table_frame = ttk.LabelFrame(trade_history_tab, text="Trade History")
        trade_table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.trade_history_tree = ttk.Treeview(
            trade_table_frame,
            columns=("timestamp", "action", "price", "size", "outcome"),
            show="headings",
        )
        self.trade_history_tree.heading("timestamp", text="Timestamp")
        self.trade_history_tree.heading("action", text="Action")
        self.trade_history_tree.heading("price", text="Price")
        self.trade_history_tree.heading("size", text="Size")
        self.trade_history_tree.heading("outcome", text="Outcome")
        self.trade_history_tree.pack(fill=tk.BOTH, expand=True)

        # Settings content
        settings_frame = ttk.LabelFrame(settings_tab, text="Bot Settings")
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(settings_frame, text="API Key:").pack(pady=5)
        self.api_key_entry = ttk.Entry(settings_frame, show="*")
        self.api_key_entry.pack(pady=5)

        ttk.Label(settings_frame, text="API Secret:").pack(pady=5)
        self.api_secret_entry = ttk.Entry(settings_frame, show="*")
        self.api_secret_entry.pack(pady=5)

        ttk.Label(settings_frame, text="Trading Pair:").pack(pady=5)
        self.pair_entry = ttk.Entry(settings_frame)
        self.pair_entry.insert(0, self.pair)
        self.pair_entry.pack(pady=5)

        ttk.Label(settings_frame, text="Trade Amount:").pack(pady=5)
        self.trade_amount_entry = ttk.Entry(settings_frame)
        self.trade_amount_entry.insert(0, str(self.trade_amount))
        self.trade_amount_entry.pack(pady=5)

        ttk.Button(settings_frame, text="Save Settings", command=self.save_settings).pack(pady=10)

    def save_settings(self):
        try:
            self.pair = self.pair_entry.get()
            self.trade_amount = float(self.trade_amount_entry.get())
            self.api.key = self.api_key_entry.get()
            self.api.secret = self.api_secret_entry.get()
            self.update_ui_log("Settings updated successfully!")
        except ValueError:
            messagebox.showerror("Invalid Input", "Ensure all fields are correctly filled.")

    def update_ui_market(self, condition):
        self.market_label.config(text=f"Market Condition: {condition}")

    def update_ui_log(self, message):
        logging.info(message)

    def update_ui_chart(self, df):
        self.ax.clear()
        self.ax.plot(df['time'], df['close'], label="Close Price")
        self.ax.set_title(f"{self.pair} Price Chart")
        self.ax.set_xlabel("Time")
       

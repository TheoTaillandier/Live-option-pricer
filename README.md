# Real-Time Option Pricer for Commodities

This project is an interactive Python application for real-time option pricing on commodities.  
It features live implied volatility, real-time risk-free rate selection depending on the Exchange, and dynamic volatility skew visualization.

## Features

- **Real-time option pricing** for major commodity markets (Grains & Oilseeds, Energy, Metals, Softs)
- **Live implied volatility** and **volatility skew** retrieval
- **Automatic risk-free rate selection** based on the exchange (US, France, Canada..)
- **User-friendly interface** with intuitive category, asset, maturity month, and year selection
- **Dynamic plotting** of volatility skew and spot price

## How it works

- The user selects a commodity category, an asset (with its exchange), the expiry month, and year. You'll have to know
- The app fetches live implied volatility and the relevant risk-free rate (US Treasury, France OAT, or Canada bond yield).
- The option price is computed in real time, and the volatility skew is displayed.

*Please note*: You'll need to know the available listed contracts for each asset in order to make the correct selection.

## Example
![image](https://github.com/user-attachments/assets/0ea96863-aa41-4717-9461-55f571d3d857)
![image](https://github.com/user-attachments/assets/a60c2e37-0526-497a-88b9-d26be4d2de8b)
![image](https://github.com/user-attachments/assets/bb492f2c-9f44-467f-8694-fceef06f43cd)





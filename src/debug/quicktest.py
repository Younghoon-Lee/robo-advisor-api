# Debugging script [ purpose for mathematical error ]

# 해당 스크립트는 디버깅을 원할하게 도와줌. 
# 에러 센트리에서 오류가 발생할 경우 해당 스크립트를 통해 debugging 후 hotfix 하는 것을 추천
# 클라우드 워치에서 로그를 확인한 후 argument에 오류를 발생시킨 값을 넣어서 test


import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns
from pypfopt.exceptions import OptimizationError
from pypfopt.efficient_frontier import EfficientFrontier
from datetime import datetime, timedelta
import pypfopt.objective_functions as objective_functions
import boto3
from boto3.dynamodb.conditions import Key
import math
import os


RISK_FREE_RATE=0.02


class SharpeRatioData:

    def __init__(self, arguments):
        self.userCash_ = []
        self.assets_ = []
        for i in range(len(arguments)):
            self.userCash_.append(arguments[i]["amount"])
            self.assets_.append(arguments[i]["stockId"])
        self.totalCashAmount = sum(self.userCash_)

    def _fetch_data(self):
        try:
            dynamodb = boto3.resource('dynamodb')
            table = dynamodb.Table(os.environ['chartTable'])
            stockEndDate = "chart-lastPrice-" + datetime.today().strftime("%Y-%m%d") + \
                "T15:00:00Z"
            stockStartDate = 'chart-lastPrice-' + \
                (datetime.today()-timedelta(weeks=52)
                 ).strftime("%Y-%m-%d")+"T15:00:00Z"

            assets = self.assets_
            df = pd.DataFrame()
            count = 0

            for asset in assets:
                if count == 0:
                    response = table.query(
                        KeyConditionExpression=Key('PK').eq(asset) & Key(
                            'SK').between(stockStartDate, stockEndDate)
                    )
                    datas = response['Items']
                    AdjPrice = []
                    dates = []
                    for data in datas:
                        AdjPrice.append(float(data['price']))
                        dates.append(data['tradeDate'])
                    df[asset] = AdjPrice
                    df['Dates'] = dates
                    df = df.sort_values(by=['Dates'])
                    df = df.set_index('Dates', drop=True)
                    count += 1
                else:
                    temp = pd.DataFrame()
                    response = table.query(
                        KeyConditionExpression=Key('PK').eq(asset) & Key(
                            'SK').between(stockStartDate, stockEndDate)
                    )
                    datas = response['Items']
                    AdjPrice = []
                    dates = []
                    for data in datas:
                        AdjPrice.append(float(data['price']))
                        dates.append(data['tradeDate'])
                    temp[asset] = AdjPrice
                    temp['Dates'] = dates
                    temp = temp.sort_values(by=['Dates'])
                    temp = temp.set_index('Dates', drop=True)
                    df = pd.merge(df, temp, how='outer',
                                  left_index=True, right_index=True)
        except Exception as e:
            print("DynamoDB connection Error :", e)
        try:
            df = df.dropna()
            self.df = df
            self.userWeights = [x/sum(self.userCash_) for x in self.userCash_]

            mu = expected_returns.capm_return(df)
            self.expectedReturnsMean = mu

            # provide methods for computing shrinkage estimates of the covariance matrix
            S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
            self.covariance = S
            ef = EfficientFrontier(mu, S)

            # increase the number of nonzero weights
            # ef.add_objective(objective_functions.L2_reg, gamma=1)

            # weights = ef.efficient_return(0.2)
            # weights = ef.clean_weights()
            # count = list(weights.values()).count(0)
            weights = ef.max_sharpe()
            # if count > 1:
            #     print("max_sharpe not applied")
            #     ef.add_objective(objective_functions.L2_reg, gamma=1)
            #     weights = ef.efficient_return(0.2)
            #     weights = ef.clean_weights()

            optimalWeights = []
            for i in range(len(assets)):
                optimalWeights.append(weights[assets[i]])
            optimalWeights = np.array(optimalWeights)
            self.optimalWeights = optimalWeights
        except OptimizationError:
            ef = EfficientFrontier(mu, S)
            weights = ef.min_volatility()
            optimalWeights = []
            for i in range(len(assets)):
                optimalWeights.append(weights[assets[i]])
            optimalWeights = np.array(optimalWeights)
            self.optimalWeights = optimalWeights
        except Exception as e:
            print("processing Data error : ", e)

    def sharpeRatioReports(self):
        try:
            S = self.covariance
            df = self.df
            cash = 100000
            userWeights = self.userWeights

            optimalWeights = self.optimalWeights

            userWeights = np.array(userWeights)

            optAssignedCash = optimalWeights * cash
            optUnitHoldings = np.divide(optAssignedCash, df.iloc[0])

            userAssignedCash = userWeights * cash
            userUnitHoldings = np.divide(userAssignedCash, df.iloc[0])

            date = []
            userPort = []
            optPort = []
            for i in range(len(df)):
                date.append(df.iloc[i].name)
                userPort.append(np.dot(userUnitHoldings, df.iloc[i])/cash)
                optPort.append(np.dot(optUnitHoldings, df.iloc[i])/cash)

            comparedResult = pd.DataFrame(
                {"Date": date, "User": userPort, "Opt": optPort})
            comparedResult = comparedResult.set_index(['Date'])
            self.comparedResult = comparedResult

            expectedReturnsMean = np.array(self.expectedReturnsMean)

            userPortfolioReturns = np.sum(expectedReturnsMean * userWeights)
            optimizedPortfolioReturns = np.sum(
                expectedReturnsMean * optimalWeights)

            userVolatility = np.sqrt(
                np.dot(userWeights.T, np.dot(S, userWeights)))
            optimalVolatility = np.sqrt(
                np.dot(optimalWeights.T, np.dot(S, optimalWeights)))
            print("유저 변동성: {}, 최적 변동성: {}".format(userVolatility, optimalVolatility))
            userPortfolioSharpe = (userPortfolioReturns-RISK_FREE_RATE)/userVolatility
            optimalPortfolioSharpe = (optimizedPortfolioReturns-RISK_FREE_RATE)/optimalVolatility
            print("유저 샤프지수: {}, 최적의 샤프지수: {}".format(userPortfolioSharpe,optimalPortfolioSharpe))

            result = {"inputPortfolio": {"expectedReturnRates": [], "volatility": 0.0, "sharpeRatio": 0.0},
                      "optimalPortfolio": {"expectedReturnRates": [], "volatility": 0.0, "sharpeRatio": 0.0}}
            for i in range(len(df)):
                result["inputPortfolio"]["expectedReturnRates"].append({"date": list(comparedResult.index)[
                    i], "expectedReturnRate": list(comparedResult["User"])[i]})
                result["optimalPortfolio"]["expectedReturnRates"].append({"date": list(comparedResult.index)[
                    i], "expectedReturnRate": list(comparedResult["Opt"])[i]})

            result['inputPortfolio']['volatility'] = userVolatility
            result['optimalPortfolio']['volatility'] = optimalVolatility
            result['inputPortfolio']['sharpeRatio'] = userPortfolioSharpe
            result['optimalPortfolio']['sharpeRatio'] = optimalPortfolioSharpe
            return result
        except Exception as e:
            print("Python Lambda Function Error : ", e)
            return {'Python Lambda Function complete': False}

    def sharpeRecommendedHoldings(self):
        try:
            df = self.df
            cash = self.totalCashAmount
            optimalWeights = self.optimalWeights
            print("target weights: {}".format(optimalWeights))
            dynamodb = boto3.resource('dynamodb')
            table = dynamodb.Table(os.environ['tickleTable'])
            response = table.query(
                KeyConditionExpression=Key('PK').eq(
                    'exchangeRate') & Key('SK').eq('currency-USD')
            )
            datas = response['Items']
            exchangeRate = float(datas[0]['rate'])
            allocatedCash = [((cash*x)/exchangeRate) for x in optimalWeights]
            latest = df.tail(1).to_numpy()
            trade_cost = latest * 0.0025
            holdings = []
            for i in range(len(latest[0])):
                if trade_cost[0][i] > 0.01:
                    holdings.append(
                        allocatedCash[i]/(latest[0][i]*1.0025*1.11*1.1))
                else:
                    holdings.append(
                        allocatedCash[i]/((latest[0][i]+0.01)*1.11*1.1))

            print("holdings before round: ", holdings)
            holdings = [(math.floor(x*100)/100) for x in holdings]
            print("holdings after round: {}".format(holdings))
            recommendedCash = []
            for i in range(len(latest[0])):
                if trade_cost[0][i] > 0.01:
                    recommendedCash.append(
                        math.ceil(holdings[i]*latest[0][i]*1.0025*1.11*1.1*exchangeRate/100)*100)
                else:
                    recommendedCash.append(
                        math.ceil(holdings[i]*(latest[0][i]+0.01)*1.1*1.11*exchangeRate/100)*100)
            print(recommendedCash)
            recommendedCash.append(cash - sum(recommendedCash))

            sharpePortfolio = {"items": []}
            for i in range(len(self.assets_)):
                sharpePortfolio["items"].append(
                    {"amount": recommendedCash[i], "stockId": self.assets_[i]})

            sharpePortfolio["items"].append(
                {"amount": recommendedCash[-1], "stockId": "cash-KRW"})
            return sharpePortfolio

        except Exception as e:
            print("Python Lambda Function Error : ", e)
            return {'python lambda function complete': False}

    def showChart(self):
        # local test for graph analysis
        comparedResult = self.comparedResult
        plt.figure(figsize=[12.2, 4.5])
        for compare in comparedResult:
            plt.plot(comparedResult[compare], label=compare)
        plt.title('Back test')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('portfolio Return Rate', fontsize=18)
        plt.legend(comparedResult.columns.values, loc='upper left')
        plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    os.environ['chartTable'] = "prod-charts"
    os.environ['tickleTable'] = 'prod-tickle'

    args = [
        {
            "amount": 5700,
            "stockId": "CH0334081137"
        },
        {
            "amount": 6000,
            "stockId": "US91332U1016"
        },
        {
            "amount": 5700,
            "stockId": "US0090661010"
        }

    ]
    test = SharpeRatioData(args)
    test._fetch_data()
    test.sharpeRatioReports()
    test.showChart()
    # holdings = test.sharpeRecommendedHoldings()
    # print(holdings)
    # print(holdings)

import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.exceptions import OptimizationError
from datetime import datetime, timedelta
import pypfopt.objective_functions as objective_functions
import boto3
import os
from boto3.dynamodb.conditions import Key
from libs.errorHandler import set_log

logger = set_log(__name__)


class SharpeRatioData:

    def __init__(self, portfolio):
        self.holdings = portfolio["items"]
        self.userCash_, self.assets_, self.defaultAmount, self.unitAmount = [], [], [], []

        for i in range(len(self.holdings)):
            self.userCash_.append(self.holdings[i]["amount"])
            self.assets_.append(self.holdings[i]["stockId"])
            self.defaultAmount.append(self.holdings[i]["defaultAmount"])
            self.unitAmount.append(self.holdings[i]["unitAmount"])
        self.totalCashAmount = sum(self.userCash_) + portfolio["cashAmount"]
        self._fetch_data()

    def _fetch_data(self):
        try:
            dynamodb = boto3.resource('dynamodb')
            table = dynamodb.Table(os.environ['chartTable'])
            stockEndDate = "chart-lastPrice-" + datetime.utcnow().strftime("%Y-%m%d") + \
                "T15:00:00Z"
            stockStartDate = 'chart-lastPrice-' + \
                (datetime.utcnow()-timedelta(weeks=52)
                 ).strftime("%Y-%m-%d")+"T15:00:00Z"

            assets = self.assets_
            df = pd.DataFrame()

            for index, asset in enumerate(assets):
                if index == 0:
                    response = table.query(
                        KeyConditionExpression=Key('PK').eq(asset) & Key(
                            'SK').between(stockStartDate, stockEndDate)
                    )
                    datas = response['Items']
                    AdjPrice, dates = [], []
                    for data in datas:
                        AdjPrice.append(float(data['price']))
                        dates.append(data['tradeDate'])
                    df[asset] = AdjPrice
                    df['Dates'] = dates
                    df = df.sort_values(by=['Dates'])
                    df = df.set_index('Dates', drop=True)
                else:
                    temp = pd.DataFrame()
                    response = table.query(
                        KeyConditionExpression=Key('PK').eq(asset) & Key(
                            'SK').between(stockStartDate, stockEndDate)
                    )
                    datas = response['Items']
                    AdjPrice, dates = [], []
                    for data in datas:
                        AdjPrice.append(float(data['price']))
                        dates.append(data['tradeDate'])
                    temp[asset] = AdjPrice
                    temp['Dates'] = dates
                    temp = temp.sort_values(by=['Dates'])
                    temp = temp.set_index('Dates', drop=True)
                    df = pd.merge(df, temp, how='outer',
                                  left_index=True, right_index=True)
        except Exception as error:
            logger.fatal("DYNAMODB CONNECTION ERROR")
            raise error
        try:
            df = df.dropna()
            self.df = df
            self.userWeights = [x/sum(self.userCash_) for x in self.userCash_]
            print("user weights : {}".format(self.userWeights))

            mu = expected_returns.capm_return(df)
            self.expectedReturnsMean = mu

            S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
            self.covariance = S
            ef = EfficientFrontier(mu, S)

            # ef.add_objective(objective_functions.L2_reg, gamma=1)

            weights = ef.max_sharpe()

            # count = list(weights.values()).count(0)
            # if count > 1:
            # ef.add_objective(objective_functions.L2_reg, gamma=1)
            # weights = ef.efficient_return(0.2)
            # weights = ef.clean_weights()

            optimalWeights = []
            for i in range(len(assets)):
                optimalWeights.append(weights[assets[i]])
            optimalWeights = np.array(optimalWeights)
            print("optimal weights : {}".format(optimalWeights))
            self.optimalWeights = optimalWeights
        except OptimizationError:
            logger.info("solver has been changed to minimize volatility due to infeasibility")
            ef = EfficientFrontier(mu, S)
            weights = ef.min_volatility()
            optimalWeights = []
            for i in range(len(assets)):
                optimalWeights.append(weights[assets[i]])
            optimalWeights = np.array(optimalWeights)
            print("optimal weights : {}".format(optimalWeights))
            self.optimalWeights = optimalWeights
        except Exception as error:
            logger.error("PROCESSING DATA ERROR")
            raise error

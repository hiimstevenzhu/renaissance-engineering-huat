import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict
import numpy as np
import pandas as pd
import math
# unsure about these libraries 
import copy
import random
import collections
from collections import defaultdict

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

# ALGO CODE GOES HERE:

# standardised global variables
INF = int(1e9)
empty_assets = {'AMETHYSTS': 0, 'STARFRUIT': 0}

class Trader:
    POS_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT':20}
    position = copy.deepcopy(empty_assets)
    volume_traded = copy.deepcopy(empty_assets)
    
    #starfuit cache
    starfruit_returns_cache = []
    starfruit_errors_cache = []
    starfruit_errors_terms = 5
    starfruit_returns_terms = 1
    last_prediction = 0
    last_starfruit_price = 5000
    
    def values_extract(self, order_dict: dict, buy=0):
        tot_vol = 0
        best_val = -1
        
        for ask, vol in order_dict.items():
            if not buy:
                vol *= -1 #quantities for selling are alw neg
            tot_vol += vol
            #if tot_vol > maxvol: #seems redundant to me, we sort alr, best_val is always the last entry
                #maxvol = vol
            best_val = ask
                
        return tot_vol, best_val
    
    def weighted_avg(self, obuy:dict, osell:dict):
        
        total_vol = 0
        total_price = 0
        for price, vol in obuy.items():
            total_vol += vol
            total_price += price * vol
        for price, vol in osell.items():
            total_vol += vol*-1
            total_price += price * (-vol)
        return total_price/total_vol
            
        
    
    def compute_orders_ame(self, algo_bid: int, algo_ask: int, order_depth: OrderDepth):
        # standardised
        orders: list[Order] = []
        product = 'AMETHYSTS'
        pos_lim = self.POS_LIMIT[product]
        
        outstanding_sell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        outstanding_buy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
        
        sell_vol, best_sell_price = self.values_extract(outstanding_sell)
        buy_vol, best_buy_price = self.values_extract(outstanding_buy, 1)

        cur_pos = self.position[product]
        
        
        # market taking all outstanding sells below our arbitrage price
        for ask, vol in outstanding_sell.items():
            if (ask < algo_bid) or ((cur_pos < 0  and (ask == algo_bid))) and cur_pos < pos_lim:
                order_amt = min(-vol, pos_lim - cur_pos)
                cur_pos += order_amt
                orders.append(Order(product, ask, order_amt))
        
        # making outstanding sells for the bots to trade on 
        
        # unsure of the value of this
        undercut_b = best_buy_price +1
        undercut_s = best_sell_price -1
        
        bid_price = min(undercut_b, algo_bid-1)
        ask_price = max(undercut_s, algo_ask+1)
        
        if (cur_pos < pos_lim):
            vol_tobuy = pos_lim-cur_pos
            if cur_pos < 0:
                # we have negative position, we want to stabilise inventory by making bid orders
                orders.append(Order(product, min(undercut_b+1, algo_bid-1), vol_tobuy)) # this line is different - we dont further undercut our buys
            elif cur_pos > 15:
                orders.append(Order(product, min(undercut_b-1, algo_bid-1), vol_tobuy))
            else:
                orders.append(Order(product, bid_price, vol_tobuy))
            cur_pos += vol_tobuy
        
        cur_pos = self.position[product]
        
        # market taking all outstanding buys above our arbitrage price
        for bid, vol in outstanding_buy.items():
            if (bid > algo_ask) or (cur_pos > 0 and (bid == algo_ask)) and cur_pos > -pos_lim:
                order_amt = max(-vol, -pos_lim - cur_pos)
                cur_pos += order_amt
                orders.append(Order(product, bid, order_amt))
        
        if (cur_pos > -pos_lim):
            vol_tosell = -pos_lim-cur_pos
            if cur_pos > 0:
                orders.append(Order(product, max(undercut_s-1, algo_ask+1), vol_tosell))
            elif cur_pos < -15:
                orders.append(Order(product, max(undercut_s+1, algo_ask+1), vol_tosell))
            else:
                orders.append(Order(product, ask_price, vol_tosell))    
                
        return orders
    
    def ar_starfruit_next_return(self):
        
        coef = [-0.3735]
        intercept = 1
        next_return = intercept
        for i, val in enumerate(self.starfruit_returns_cache):
            next_return += val * coef[i]

        coefMA = [0.0126,-0.034,-0.0057,-0.2645,-0.3344]
        interceptMA = 0.0000000754
        next_return += interceptMA
        for i, val in enumerate(self.starfruit_errors_cache):
            next_return += val * coefMA[i]
        return next_return
    
    def compute_orders_star(self, algo_bid: int, algo_ask: int, order_depth: OrderDepth):
        # standardised
        orders: list[Order] = []
        product = 'STARFRUIT'
        pos_lim = self.POS_LIMIT[product]
        
        outstanding_sell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        outstanding_buy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
        
        sell_vol, best_sell_price = self.values_extract(outstanding_sell)
        buy_vol, best_buy_price = self.values_extract(outstanding_buy, 1)

        cur_pos = self.position[product]
        
        # market take outstanding sells
        for ask, vol in outstanding_sell.items():
            if (ask < algo_bid) or ((cur_pos < 0  and (ask == algo_bid))) and cur_pos < pos_lim:
                order_amt = min(-vol, pos_lim - cur_pos)
                cur_pos += order_amt
                orders.append(Order(product, ask, order_amt))
                
        # undercutting for bids and asks
        undercut_b = best_buy_price +1
        undercut_s = best_sell_price -1
        
        bid_price = min(undercut_b, algo_bid-1)
        ask_price = max(undercut_s, algo_ask+1)
        
        if cur_pos < pos_lim:
            order_amt = pos_lim - cur_pos
            orders.append(Order(product, bid_price, order_amt))
            cur_pos += order_amt
        
        cur_pos = self.position[product]
            
        # market take outstanding buys
        for bid, vol in outstanding_buy.items():
            if (bid > algo_ask) or (cur_pos > 0 and (bid == algo_ask)) and cur_pos > -pos_lim:
                order_amt = max(-vol, -pos_lim - cur_pos)
                cur_pos += order_amt
                orders.append(Order(product, bid, order_amt))
        
        if cur_pos > -pos_lim:
            order_amt = -pos_lim-cur_pos
            orders.append(Order(product, ask_price, order_amt))
            cur_pos += order_amt
                
        return orders
        
        
        
        
    def run(self, state: TradingState):        
        # base requirements
        result = {'AMETHYSTS': [], 'STARFRUIT': []}
        # We iterate through keys in the order depth to update algo's position in an asset
        for key, val in state.position.items():
            self.position[key] = val
            print(f'{key} position: {val}')
        print()
        
        timestamp = state.timestamp
        # AMETHYSTS tend to be stable - we'll just implement simple market-making
        ame_lb = ame_ub = 10000
        ame_orders = self.compute_orders_ame(ame_lb, ame_ub, state.order_depths['AMETHYSTS'])
        result['AMETHYSTS'] = ame_orders
        
        
        
        # STARFRUITS
        # we keep the last 3 prices
        
        if len(self.starfruit_returns_cache) == self.starfruit_returns_terms:
            self.starfruit_returns_cache.pop(0)
            
        #s_vol, best_sell_star = self.values_extract(collections.OrderedDict(sorted(state.order_depths["STARFRUIT"].sell_orders.items(), reverse=True)))
        #b_vol, best_buy_star = self.values_extract(collections.OrderedDict(sorted(state.order_depths["STARFRUIT"].buy_orders.items())), 1)
        lowest_asking = 5000
        biggest_bid = 5000
        if len(state.order_depths["STARFRUIT"].sell_orders.items()) != 0:
            lowest_asking = INF
            for price, vol in collections.OrderedDict(sorted(state.order_depths["STARFRUIT"].sell_orders.items())).items():
                lowest_asking = min(lowest_asking, price)
        if len(state.order_depths["STARFRUIT"].buy_orders.items()) != 0:
            biggest_bid = -INF
            for price, vol in collections.OrderedDict(sorted(state.order_depths["STARFRUIT"].buy_orders.items())).items():
                biggest_bid = max(biggest_bid, price)
        #        lowest_asking, best_ask_amount = list(state.order_depths["STARFRUIT"].sell_orders.items())[0]
        #if len(state.order_depth.buy_orders) != 0:
        #        biggest_bid, best_bid_amount = list(state.order_depths["STARFRUIT"].buy_orders.items())[0]

        
        mid_price = ( lowest_asking + biggest_bid )/2
        #mid_price = self.weighted_avg(collections.OrderedDict(sorted(state.order_depths["STARFRUIT"].buy_orders.items())), collections.OrderedDict(sorted(state.order_depths["STARFRUIT"].sell_orders.items())))
        cur_returns = mid_price/self.last_starfruit_price
        # replace the saved price for return calc as prev mid
        self.last_starfruit_price = mid_price
        self.starfruit_returns_cache.append(cur_returns)
        star_lb = -INF
        star_ub = INF
        '''
        if len(self.starfruit_cache) == self.starfruit_terms:
            star_next_price = self.ar_starfruit()
            star_lb = star_next_price-1
            star_ub = star_next_price+1
        '''
        if self.last_prediction != 0:
            error = cur_returns - self.last_prediction
            self.starfruit_errors_cache.append(error)
            if len(self.starfruit_errors_cache) > self.starfruit_errors_terms:
                self.starfruit_errors_cache.pop()

        if (len(self.starfruit_errors_cache) == self.starfruit_errors_terms) and (len(self.starfruit_returns_cache)== self.starfruit_returns_terms):
            star_next_return = self.ar_starfruit_next_return()
            self.last_prediction = star_next_return
            star_lb = int(round((mid_price*star_next_return) -1))
            star_ub = int(round((mid_price*star_next_return) +1))
            
            
        star_orders = self.compute_orders_star(star_lb, star_ub, state.order_depths["STARFRUIT"])
        result["STARFRUIT"] = star_orders
            
		# String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        logger.flush(state, result, None, traderData)
        return result, None, traderData
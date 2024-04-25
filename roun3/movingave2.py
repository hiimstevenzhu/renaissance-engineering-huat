# ROUND 1 BENCHMARK

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
empty_assets = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0, 'GIFT_BASKET': 0}

class Trader:
    POS_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT':20, 'ORCHIDS':100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60, 'GIFT_BASKET': 60}
    position = copy.deepcopy(empty_assets)
    volume_traded = copy.deepcopy(empty_assets)
    
    #starfuit cache
    starfruit_cache = []
    starfruit_terms = 4

    #choc cache
    choc_cache = []
    choc_terms = 21

    rs_cache = []
    rs_terms = 21

    gb_cache = []
    gb_terms = 21

    sb_cache = []
    sb_terms = 21
    
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
    
    def ar_starfruit(self):
        coef = [0.1921, 0.1957, 0.2627, 0.3461]
        intercept = 17.3638 
        next_price = intercept
        for i, val in enumerate(self.starfruit_cache):
            next_price += val * coef[i]
        return int(round(next_price))
    
    def lr_orchid(self, observation: Observation):
        orc_ask_price = observation.askPrice
        orc_bid_price = observation.bidPrice
        # orc_mid_price = (orc_ask_price+orc_bid_price)/2
        # humidity = observation.humidity
        # sunlight = observation.sunlight
        
        # # calculate humidity coef
        # if 60 <= humidity <= 80:
        #     hum_coef = 1
        # else:
        #     deviation = min(abs(humidity-60), abs(humidity-80))
        #     drop = deviation/5 * 2
        #     hum_coef = 1 * (100-drop)/100
        
        # # calculate sunlight coef
        # sun_coef = 1 + (2500-sunlight)/2500
        
        # coef = [-0.34070384, -0.04198594,  0.99980238] #hum_coef, sun_coef, cur_price
        # intercept = 0.585799733940803
        # next_mid_price = intercept + hum_coef*coef[0] + sun_coef*coef[1] + orc_mid_price*coef[2]
        # # TEST LINE
        # next_mid_price = round(next_mid_price*4)/4
        # next_bid_price = next_mid_price-0.75
        # next_ask_price = next_mid_price+0.75

        return (orc_bid_price, orc_ask_price)
    
    def compute_orders_regression(self, algo_bid: int, algo_ask: int, state: TradingState, product: str):
        # standardised
        order_depth = state.order_depths[product]
        orders: list[Order] = []
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
    
    def compute_orders_orchid(self, algo_bid: int, algo_ask: int, state: TradingState, product: str, observation: Observation):
        # standardised
        order_depth = state.order_depths[product]
        orders: list[Order] = []
        pos_lim = self.POS_LIMIT[product]
        
        outstanding_sell = collections.OrderedDict(sorted(order_depth.sell_orders.items(), reverse=True))
        outstanding_buy = collections.OrderedDict(sorted(order_depth.buy_orders.items()))
        
        sell_vol, best_sell_price = self.values_extract(outstanding_sell)
        buy_vol, best_buy_price = self.values_extract(outstanding_buy, 1)
        
        undercut_bid = best_buy_price + 0 #SET AT 0 UNTIL MORE INFO OUT
        
        diff = (best_sell_price - best_buy_price) * 0.232
        
        undercut_ask = algo_ask + observation.importTariff + observation.transportFees + diff # diff

        
        cur_pos = self.position[product]
        # just making asks
        algo_ask = algo_ask
        order_amt = -100
        orders.append(Order(product, int(round(undercut_ask)), order_amt))
        
                
        return orders, cur_pos

    def compute_orders_gift(self, mid_price_choc, order_depth_choc: OrderDepth, order_depth_Basket: OrderDepth):
        orders: list[Order] = []

        for i, val in enumerate(self.choc_cache):
            i+=1
            if i < 3:
                prev_price = val
            else:
                break
                
        outstanding_sell_basket = collections.OrderedDict(sorted(order_depth_Basket.sell_orders.items()))
        outstanding_buy_basket = collections.OrderedDict(sorted(order_depth_Basket.buy_orders.items(), reverse=True))
        
        best_sell_price_basket = next(iter(outstanding_sell_basket))
        best_buy_price_basket = next(iter(outstanding_buy_basket))
        mid_price_basket = (best_sell_price_basket + best_buy_price_basket) / 2
            
        ask_price = int(math.floor(mid_price_basket))

        if mid_price_choc < prev_price:
            orders.append(Order("GIFT_BASKET", ask_price, -60))
        
        elif mid_price_choc > prev_price: 
            orders.append(Order("GIFT_BASKET", ask_price, 60))

        return orders

    def moving_average_strategy(self, prices_list, symbol='STOCK', short_window=5, long_window=20):
        prices = pd.Series(prices_list)
        
        short_moving_avg = prices.rolling(window=short_window).mean()
        long_moving_avg = prices.rolling(window=long_window).mean()

        signals = pd.DataFrame(index=prices.index)
        signals['signal'] = 0.0
        signals['short_mavg'] = short_moving_avg
        signals['long_mavg'] = long_moving_avg
        signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)   
        signals['positions'] = signals['signal'].diff()

        # Initialize an empty list to store orders
        orders: list[Order] = []

        pos_lim = self.POS_LIMIT[symbol]

        # Generate orders based on the signals
        for index, signal in signals.iterrows():
            # A buy signal (positions changes from 0 to 1)
            if signal['positions'] == 1.0:
                # Buy order with the last known price and some hypothetical quantity
                orders.append(Order(symbol, int(math.floor(prices_list[index])), pos_lim))  # Example quantity
            # A sell signal (positions changes from 1 to 0)
            elif signal['positions'] == -1.0:
                # Sell order with the last known price and some hypothetical quantity
                orders.append(Order(symbol, int(math.floor(prices_list[index])), -pos_lim))  # Example quantity

        return orders

    def run(self, state: TradingState):        
        # base requirements
        result = {'AMETHYSTS': [], 'STARFRUIT': [], 'ORCHIDS': []}
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
        if len(self.starfruit_cache) == self.starfruit_terms:
            self.starfruit_cache.pop(0)
        s_vol, best_sell_star = self.values_extract(collections.OrderedDict(sorted(state.order_depths["STARFRUIT"].sell_orders.items())))
        b_vol, best_buy_star = self.values_extract(collections.OrderedDict(sorted(state.order_depths["STARFRUIT"].buy_orders.items(), reverse=True)), 1)
        self.starfruit_cache.append((best_buy_star+best_sell_star) / 2)
        star_lb = -INF
        star_ub = INF
        if len(self.starfruit_cache) == self.starfruit_terms:
            star_next_price = self.ar_starfruit()
            star_lb = star_next_price-1
            star_ub = star_next_price+1
        star_orders = self.compute_orders_regression(star_lb, star_ub, state, "STARFRUIT")
        result["STARFRUIT"] = star_orders

        # ORCHIDS
        orc_next_bid, orc_next_ask = self.lr_orchid(state.observations.conversionObservations["ORCHIDS"])
        orc_lb = orc_next_bid
        orc_ub = orc_next_ask 
        orc_orders, orchid_pos = self.compute_orders_orchid(orc_lb, orc_ub, state, "ORCHIDS", state.observations.conversionObservations["ORCHIDS"])
        result["ORCHIDS"] = orc_orders
        
        
        orchid_pos = self.position['ORCHIDS']
        conversions = -orchid_pos


        #choc_gift lead lag strategy
        # if len(self.choc_cache) == self.choc_terms:
        #     self.choc_cache.pop(0)

        # outstanding_sell_choc = collections.OrderedDict(sorted(state.order_depths["CHOCOLATE"].sell_orders.items(), reverse=True))
        # outstanding_buy_choc = collections.OrderedDict(sorted(state.order_depths["CHOCOLATE"].buy_orders.items()))
        
        # best_sell_price_choc = next(iter(outstanding_sell_choc))
        # best_buy_price_choc = next(iter(outstanding_buy_choc))
        # mid_price_choc = (best_sell_price_choc + best_buy_price_choc) / 2
        # self.choc_cache.append(mid_price_choc)

        # gift_order = self.compute_orders_gift(mid_price_choc, state.order_depths['CHOCOLATE'], state.order_depths['GIFT_BASKET'])
        # result['GIFT_BASKET'] = gift_order    

        if len(self.gb_cache) == self.gb_terms:
            self.gb_cache.pop(0)
        outstanding_sell_gb = collections.OrderedDict(sorted(state.order_depths["CHOCOLATE"].sell_orders.items(), reverse=True))
        outstanding_buy_gb = collections.OrderedDict(sorted(state.order_depths["CHOCOLATE"].buy_orders.items()))
        
        best_sell_price_gb = next(iter(outstanding_sell_gb))
        best_buy_price_gb = next(iter(outstanding_buy_gb))
        mid_price_gb = (best_sell_price_gb + best_buy_price_gb) / 2
        self.gb_cache.append(mid_price_gb)

        if len(self.choc_cache) == self.choc_terms:
            self.choc_cache.pop(0)
        outstanding_sell_choc = collections.OrderedDict(sorted(state.order_depths["CHOCOLATE"].sell_orders.items(), reverse=True))
        outstanding_buy_choc = collections.OrderedDict(sorted(state.order_depths["CHOCOLATE"].buy_orders.items()))
        
        best_sell_price_choc = next(iter(outstanding_sell_choc))
        best_buy_price_choc = next(iter(outstanding_buy_choc))
        mid_price_choc = (best_sell_price_choc + best_buy_price_choc) / 2
        self.choc_cache.append(mid_price_choc)

        if len(self.sb_cache) == self.sb_terms:
            self.sb_cache.pop(0)
        outstanding_sell_sb = collections.OrderedDict(sorted(state.order_depths["CHOCOLATE"].sell_orders.items(), reverse=True))
        outstanding_buy_sb = collections.OrderedDict(sorted(state.order_depths["CHOCOLATE"].buy_orders.items()))
        
        best_sell_price_sb = next(iter(outstanding_sell_sb))
        best_buy_price_sb = next(iter(outstanding_buy_sb))
        mid_price_sb = (best_sell_price_sb + best_buy_price_sb) / 2
        self.sb_cache.append(mid_price_sb)

        if len(self.rs_cache) == self.rs_terms:
            self.rs_cache.pop(0)
        outstanding_sell_rs = collections.OrderedDict(sorted(state.order_depths["CHOCOLATE"].sell_orders.items(), reverse=True))
        outstanding_buy_rs = collections.OrderedDict(sorted(state.order_depths["CHOCOLATE"].buy_orders.items()))
        
        best_sell_price_rs = next(iter(outstanding_sell_rs))
        best_buy_price_rs = next(iter(outstanding_buy_rs))
        mid_price_rs = (best_sell_price_rs + best_buy_price_rs) / 2
        self.rs_cache.append(mid_price_rs)

        if len(self.gb_cache) == self.gb_terms:
            result['GIFT_BASKET'] = self.moving_average_strategy(self.gb_cache, 'GIFT_BASKET')
        if len(self.choc_cache) == self.choc_terms:
            result['CHOCOLATE'] = self.moving_average_strategy(self.choc_cache, 'CHOCOLATE')
        if len(self.sb_cache) == self.sb_terms:
            result['STRAWBERRIES'] = self.moving_average_strategy(self.sb_cache, 'STRAWBERRIES')
        if len(self.rs_cache) == self.rs_terms:
            result['ROSES'] = self.moving_average_strategy(self.rs_cache, 'ROSES')


		# String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
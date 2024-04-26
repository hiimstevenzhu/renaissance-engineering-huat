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
empty_assets = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0, 'GIFT_BASKET': 0, 'COCONUT': 0, 'COCONUT_COUPON': 0}

class Trader:
    POS_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT':20, 'ORCHIDS':100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60, 'GIFT_BASKET': 60, 'COCONUT': 300, 'COCONUT_COUPON': 600}
    position = copy.deepcopy(empty_assets)
    volume_traded = copy.deepcopy(empty_assets)
    
    #starfuit cache
    starfruit_cache = []
    starfruit_terms = 4

    # basket z-score cache
    prev_basket_zscore = 0
    hold_gb = 0
    rihanna_rose = None


    # coconuts cache
    day = 3
    coc_vol = 0.0001013965
    
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
        coef = [0.1890, 0.2077, 0.2611, 0.3418]
        intercept = 2.3565 
        next_price = intercept
        for i, val in enumerate(self.starfruit_cache):
            next_price += val * coef[i]
        return int(round(next_price))
    
    def lr_orchid(self, observation: Observation):
        orc_ask_price = observation.askPrice
        orc_bid_price = observation.bidPrice
        return (orc_bid_price, orc_ask_price)
 
        
    
    def compute_orders_regression(self, algo_bid: int, algo_ask: int, state: TradingState, product: str):
        # standardised
        adam_bid = 0
        adam_ask = 0
     
        order_depth = state.order_depths[product]
        orders: list[Order] = []
        pos_lim = self.POS_LIMIT[product]
        
        outstanding_sell = collections.OrderedDict(sorted(order_depth.sell_orders.items(), reverse=True))
        outstanding_buy = collections.OrderedDict(sorted(order_depth.buy_orders.items()))
        
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
        undercut_b = best_buy_price 
        undercut_s = best_sell_price 
        

        
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
        
        outstanding_sell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        outstanding_buy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
        
        sell_vol, best_sell_price = self.values_extract(outstanding_sell)
        buy_vol, best_buy_price = self.values_extract(outstanding_buy, 1)
        
        storage_price = 0.1
        
        # initialising the standard ask and bid for market making
        diff = (best_sell_price - best_buy_price) * 0.5
        mm_bid = observation.bidPrice + 1.5
        mm_ask = observation.askPrice - 1.5 # diff
        
        # conditional to check which direction should be taken
        # we max out on the short when we find tht selling at a confident price has the highest profits per orchid
        # and we max out on the long when we find tht buying at a confident price has the highest profits per orchid
        mid_price = (best_sell_price + best_buy_price) * 0.5
        long_profitability = observation.askPrice - observation.exportTariff - observation.transportFees - mm_bid #profit we get if we buy from local and sell to south
        short_profitability = mm_ask -  observation.importTariff - observation.transportFees - observation.bidPrice #baseline value we are okay with selling to local and buying from south
        
        condition = long_profitability > short_profitability # if we earn more from longing, go long
        
        take_aggression = 1
        profitable_bid = observation.askPrice - observation.exportTariff - observation.transportFees - 2.5
        profitable_ask = observation.bidPrice +  observation.importTariff + observation.transportFees + 2.5
        

        cur_pos = self.position[product]
        
        if condition:
            # for ask, vol in outstanding_sell.items():
            #     if (ask < profitable_bid) and cur_pos < pos_lim:
            #         order_amt = min(vol, pos_lim - cur_pos)
            #         cur_pos += order_amt
            #         orders.append(Order(product, ask, order_amt))
            # market make to long orchids locally on high volume
            if long_profitability > 1.5:
                order_amt = self.POS_LIMIT['ORCHIDS'] - cur_pos
                orders.append(Order(product, int(round(mm_bid)), order_amt))
        
        else:
            # for bid, vol in outstanding_buy.items():
            #     if (bid > profitable_ask) and cur_pos > -pos_lim:
            #         order_amt = max(-vol, -pos_lim - cur_pos)
            #         cur_pos += order_amt
            #         orders.append(Order(product, bid, order_amt))
            if short_profitability > 1.5:
                order_amt = -self.POS_LIMIT[product] - cur_pos
                orders.append(Order(product, int(round(mm_ask)), order_amt))
                
        return orders
    
    def compute_orders_basket(self, order_depth: OrderDepth, state: TradingState):

        if 'ROSES' in state.market_trades:
                for i in state.market_trades['ROSES']:
                    if i.buyer == 'Rhianna':
                        self.rihanna_rose = 'LONG' 
                        print('RIHANNA LONG')
                    if i.seller == 'Rhianna':
                        self.rihanna_rose = 'SHORT'
                        print('RIHANNA SHORT')

        mean_difference = 379.5
        standard_dev = 76.4
        straw = 'STRAWBERRIES'
        choc = 'CHOCOLATE'
        roses = 'ROSES'
        gb = 'GIFT_BASKET'
        orders = {straw: [], choc: [], roses: [], gb: []}
        products = ['STRAWBERRIES', 'CHOCOLATE', 'ROSES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}
        
        # initialising values
        for p in products:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))
            mid_price[p] = (best_sell[p] + best_buy[p])/2

            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
                if vol_buy[p] >= self.POS_LIMIT[p]/10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                if vol_sell[p] >= self.POS_LIMIT[p]/10:
                    break

        trade_at = standard_dev*0.8

        price_diff = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['STRAWBERRIES']*6 - mid_price['ROSES'] - mean_difference
        if abs(price_diff) > trade_at and abs(price_diff) <= trade_at*1.4:
            take_pos = round(45 + 15 * (abs(price_diff) - trade_at)/(trade_at*0.6))
        elif abs(price_diff) > trade_at*1.4:
            take_pos = 60
       
        take_pos = 60

        if price_diff > trade_at and abs(self.position['GIFT_BASKET']) < abs(take_pos): #short gift basket, long the others!
            vol = self.position['GIFT_BASKET'] + take_pos
            #self.cont_buy_basket_unfill = 0  -- ignore i think they used this for pair trade
            # if vol > 0:
            gb_cutting = 0
            cutting = 0
            orders[gb].append(Order(gb, best_buy[gb]-1, -vol))
            # if (self.rihanna_rose == None):
            #     orders[straw].append(Order(straw, best_sell[straw]+1, vol*6))
            # orders[choc].append(Order(choc, best_sell[choc]+1, vol*4))
            # orders[roses].append(Order(roses, best_sell[roses]+1, vol))
            #orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) 
                #self.cont_sell_basket_unfill += 2
                #pb_neg -= vol

        elif price_diff < -trade_at and abs(self.position['GIFT_BASKET']) < abs(take_pos):
            vol = take_pos - self.position['GIFT_BASKET']
            #self.cont_sell_basket_unfill = 0 # no need to sell rn
            assert(vol >= 0)
            # if vol > 0:
            orders[gb].append(Order(gb, best_sell[gb]+1, vol))
            # if (self.rihanna_rose == None):
            #     orders[straw].append(Order(straw, best_buy[straw]-1, -vol*6))
            # orders[choc].append(Order(choc, best_buy[choc]-1, -vol*4))
            # orders[roses].append(Order(roses, best_buy[roses]-1, -vol))

            #orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                #self.cont_buy_basket_unfill += 2
                #pb_pos += vol

        if self.position['GIFT_BASKET'] < 0 and price_diff < -trade_at*0.25: #Currently shorted. I want to buy back my items to have 0 position
            vol = - self.position['GIFT_BASKET']
            assert(vol >= 0)
            orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol)) 
        if self.position['GIFT_BASKET'] > 0 and price_diff > trade_at*0.25: #Currently longed. I want to sell items to have 0 position
            vol = - self.position['GIFT_BASKET']
            assert(vol <= 0)
            orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], vol)) 

        if self.rihanna_rose == 'LONG':
            vol = self.POS_LIMIT['ROSES'] - self.position['ROSES']
            orders[roses].append(Order(roses, worst_sell['ROSES'], vol))
        if self.rihanna_rose == 'SHORT':
            vol = - self.POS_LIMIT['ROSES'] - self.position['ROSES']
            orders[roses].append(Order(roses, worst_buy['ROSES'], vol))
   
        
        return orders
    
    def compute_orders_coconuts(self, order_depth: OrderDepth, timestamp: int):
        premium = 637.63
        coc = 'COCONUT'
        cou = 'COCONUT_COUPON'
        orders = {coc: [], cou: []}
        products = [coc, cou]
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, positions = {}, {}, {}, {}, {}, {}, {}, {}
        # initialising values
        for p in products:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]), 10000000)
            best_buy[p] = next(iter(obuy[p]), 10000000)

            worst_sell[p] = next(reversed(osell[p]), 10000000)
            worst_buy[p] = next(reversed(obuy[p]), 10000000)
            mid_price[p] = (best_sell[p] + best_buy[p])/2
            
            positions[p] = self.position[p]
        # if timestamp == 0:
        #     self.day += 1
        
        day = timestamp/1000000 + self.day
        T = (250-day)/250
        bs_pricing = self.calc_BS(mid_price[coc], T)
        cou_deviation = mid_price[cou] - bs_pricing
        print(f'{T} years left, {bs_pricing} is predicted, {mid_price[cou]} as comparison')
        short_at = 5.5
        long_at = -5.5
        
        if abs(cou_deviation) > 5.5 and abs(cou_deviation) <= 9:
            take_pos = round(450 + 150 * (abs(cou_deviation) - 6)/3.5)
        elif abs(cou_deviation) > 8:
            take_pos = 600
       
        

        positions[cou] = self.position['COCONUT_COUPON']
        # arbitraging on dev
        if cou_deviation >= short_at and positions[cou] > -take_pos:
            # short coco couopons
            cou_diff = take_pos + positions[cou]
            orders[cou].append(Order(cou, best_sell[cou]-1, -cou_diff))

        elif cou_deviation <= long_at and positions[cou] < take_pos:
            # long coco coupons
            cou_diff = take_pos - positions[cou]
            orders[cou].append(Order(cou, best_buy[cou]+1, cou_diff))

        # #close positions at
        # elif cou_deviation >= 2.5 and positions[cou] > 0:
        #     orders[cou].append(Order(cou, best_sell[cou]-1, -positions[cou]))
        # elif cou_deviation <= -2.5 and positions[cou] < 0:
        #     orders[cou].append(Order(cou, best_buy[cou]+1, -positions[cou]))



        return orders
        
    def calc_BS(self, mid_price, T):
        vol = 0.0001013965
        sigma = vol *  np.sqrt(10000) * np.sqrt(250)
        S = mid_price
        K = 10000
        r = 0
        d1 = (np.log(S/K) + (r + (sigma**2)/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        res = S * self.NCDF(d1) - K * np.exp(-r*T)* self.NCDF(d2)
        print(S, sigma, d1, d2, res, T)
        return res

    def NCDF(self, x):
        return 0.5*(1+math.erf(x/math.sqrt(2)))
        
    def run(self, state: TradingState):        
        # base requirements
        result = {'AMETHYSTS': [], 'STARFRUIT': [], 'ORCHIDS': []}
        # We iterate through keys in the order depth to update algo's position in an asset
        for key, val in state.position.items():
            self.position[key] = val
            print(f'{key} position: {val}')
        print()
        # for back testing purposes
        conversions = self.position['ORCHIDS']
      
        timestamp = state.timestamp
        # AMETHYSTS tend to be stable - we'll just implement simple market-making
        ame_lb = ame_ub = 10000
        if 'AMETHYSTS' in state.order_depths.keys():
            ame_orders = self.compute_orders_ame(ame_lb, ame_ub, state.order_depths['AMETHYSTS'])
            result['AMETHYSTS'] = ame_orders
        
        
        
        # STARFRUITS
        # we keep the last 3 prices
        if 'STARFRUIT' in state.order_depths.keys():
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
        if 'ORCHIDS' in state.order_depths.keys():
            orc_next_bid, orc_next_ask = self.lr_orchid(state.observations.conversionObservations["ORCHIDS"])
            orc_lb = orc_next_bid # replace aggression
            orc_ub = orc_next_ask # replace aggression
            orc_orders = self.compute_orders_orchid(orc_lb, orc_ub, state, "ORCHIDS", state.observations.conversionObservations["ORCHIDS"])
            result["ORCHIDS"] = orc_orders
            conversions = self.position['ORCHIDS']
        orchid_pos = self.position['ORCHIDS']    
        conversions = -orchid_pos
        
        # BASKET GROUP - LEADING INDICATOR
        if 'GIFT_BASKET' in state.order_depths.keys() and 'ROSES' in state.order_depths.keys() and 'CHOCOLATE' in state.order_depths.keys() and 'STRAWBERRIES' in state.order_depths.keys():
            basket_orders = self.compute_orders_basket(state.order_depths, state)
            result['GIFT_BASKET'] = basket_orders['GIFT_BASKET']
            result['ROSES'] = basket_orders['ROSES']
            result['STRAWBERRIES'] = basket_orders['STRAWBERRIES']
            result['CHOCOLATE'] = basket_orders['CHOCOLATE']
            
        if 'COCONUT' in state.order_depths.keys():
            coconut_orders = self.compute_orders_coconuts(state.order_depths, timestamp)
            result['COCONUT'] = coconut_orders['COCONUT']
            result['COCONUT_COUPON'] = coconut_orders['COCONUT_COUPON']
        
        if 'AMETHYSTS' in state.market_trades:
            for i in state.market_trades['AMETHYSTS']:
                print('TRADED BY', i.buyer, i.seller)


		# String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
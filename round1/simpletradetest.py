from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            if product == "AMETHYSTS":
                order_depth: OrderDepth = state.order_depths[product]
                #print("====---> I WANT TO SEE Order depth:", order_depth)
                orders: List[Order] = []

                # Calculate the mean price from order depth
                total_price, total_count = 0, 0
                for price, amount in order_depth.sell_orders.items():
                    total_price += int(price) * amount
                    total_count += amount
                for price, amount in order_depth.buy_orders.items():
                    total_price += int(price) * amount
                    total_count += amount
                if total_count > 0:
                    mean_price = total_price / total_count
                else:
                    mean_price = 0  # Default mean price in case there are no orders

                # Set thresholds based on mean price
                sell_price_threshold = mean_price * 1.10  # 110% of mean price
                buy_price_threshold = mean_price * 0.90  # 90% of mean price

                # Process sell orders
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    if int(best_ask) < buy_price_threshold:
                        print(f"BUY AMETHYSTS {str(best_ask_amount)}x at price {best_ask}")
                        orders.append(Order("AMETHYSTS", best_ask, max(best_ask_amount, 20)))

                # Process buy orders
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    if int(best_bid) > sell_price_threshold:
                        print(f"SELL AMETHYSTS {str(best_bid_amount)}x at price {best_bid}")
                        orders.append(Order("AMETHYSTS", best_bid, min(-best_bid_amount, -20)))  # Negative amount for selling
                
                result[product] = orders
                break  # Remove this if you want to process other products as well
    
        # Trader state data
        traderData = "SAMPLE" 
        
        # Sample conversion request
        conversions = 0
        return result, conversions, traderData

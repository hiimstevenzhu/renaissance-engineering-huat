max = 0
max_x = 0
for x in range(0,101):
    cur_profits = (x/100) ** 2 * (1000-x) + ((980 / 100) ** 2 - (x/100) **2 * (1000-980) )
    if cur_profits > max:
        max = cur_profits
        max_x = x
print(max_x, cur_profits)
    
# storex = storey = maxprofit = 0
# for x in range(0,101):
#     # for y in range(x,101):
#     y = 80
#     currentprofit = (x/100)**2 * (100-x) + ((y/100)**2 -(x/100)**2)* (100-y)
#     if currentprofit > maxprofit:
#         storex = x
#         storey = y
#         maxprofit = currentprofit

# print("this is x", storex)
# print("this is y", storey)
# print("this is maxprofit", maxprofit)


maxprofits = []
ave = 81
for x in range(0, 101):
    for y in range(77,83):
        currentprofit = (x / 100) ** 2 * (100 - x) + ((y / 100) ** 2 - (x / 100) ** 2) * (100 - y) * min(1, (100-ave)/(100-y))
        maxprofits.append((currentprofit, x, y))

maxprofits.sort(reverse=True, key=lambda p: p[0])

print("Top 5 profits:")
for profit, x, y in maxprofits[:30]:
    print(f"Profit: {profit:.4f}, x: {x}, y: {y}")

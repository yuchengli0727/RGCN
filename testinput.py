import sys
'''
#for line in sys.stdin:
 #   a = line
line = sys.stdin.readline().strip().split()
line = [float(i) for i in line]


try:
    while True:
        foodnum = line[0]
    oriprice = 0
    disprice = 0
    for food in range(1, len(line) - 2):
        if food % 2 == 1:
            if line[food] < line[food + 1]:
                print("error")
                break
            oriprice += line[food]
        else:
            disprice += line[food]

    couponprice = oriprice - (oriprice // line[-2]) * line[-1]

    if couponprice > disprice:
        print(disprice)
    else:
        print(couponprice)
except:
    pass
'''
'''
c = list(map(float, input().split()))
'''


n = int(input())       

t = []
for i in range(n + 1):
    tt = input().split()
    t.append(float(tt[0]))
    t.append(float(tt[1]))


def solution(price:list):
    oriprice = 0
    disprice = 0
    for food in range(len(price) - 2):
        if food % 2 == 0:
            if price[food] < price[food + 1]:
                return -1
            oriprice += price[food]
        else:
            disprice += price[food]
    couponprice = oriprice - (oriprice // price[-2]) * price[-1]
    return min(couponprice, disprice)


result = solution(t)

if result == -1:
    print("error")
else:
    print("%.2f" % result)

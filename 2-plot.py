import matplotlib.pyplot as plt


def get_chart_data(country_code):
    f = open("./output/tp2_2_mavg_data_{}.csv/all.csv".format(country_code), 'r')

    x = []
    y = []
    for l in f:
        l = l[0:-1]
        parts = l.split(",")
        x.append(parts[0])
        y.append(float(parts[1]))

    return [x, y]


plt.title("# of tweets in US, BR and MX")
plt.xlabel("date")
plt.ylabel("# of tweets")
plt.xticks(rotation=90)
plt.grid()

line_US, = plt.plot(*get_chart_data("US"), label="US")
line_BR, = plt.plot(*get_chart_data("BR"), label="BR")
line_MX, = plt.plot(*get_chart_data("MX"), label="MX")

plt.legend(handles=[line_US, line_BR, line_MX], loc='upper right')

plt.savefig("./output/2.png")

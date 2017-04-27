import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv('device1.csv', header=0, sep=',')
df2 = pd.read_csv('device2.csv', header=0, sep=',')

t1 = pd.to_datetime(df1['time_point'].astype(str))
t2 = pd.to_datetime(df2['time_point'].astype(str))


def draw_line_chart( str ):
    d1 = df1[str]
    d2 = df2[str]
    plt.plot(t1, d1, 'b', label='device1')
    plt.plot(t2, d2, 'r', label='device2')
    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.ylabel(str)
    plt.title('Two iOS devices data on '+str)
    # plt.ylim(5, 15)
    plt.show()


def draw_accumulative_chart( str2 ):
    d1 = df1[str2]
    d2 = df2[str2]

    temp1 = d1[len(d1) - 1]
    d1_accu = [0] * len(d1)
    d1_accu[len(d1) - 1] = temp1
    for i in range(1, len(d1) - 1)[::-1]:
        value1 = d1[i] + temp1
        temp1 = value1
        d1_accu[i - 1] = value1

    temp2 = d2[len(d2)-1]
    d2_accu = [0]*len(d2)
    d2_accu[len(d2)-1] = temp2
    for i in range(1, len(d2)-1)[::-1]:
        value2 = d2[i] + temp2
        temp2 = value2
        d2_accu[i-1] = value2

    plt.plot(t1, d1_accu, 'b', label='device1')
    plt.plot(t2, d2_accu, 'r', label='device2')
    plt.xlabel('Time')
    plt.ylabel(str2)
    plt.title('Two iOS devices data on accumulative ' + str2)
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    draw_line_chart('pm25_concen')
    # draw_line_chart('outdoor')
    # draw_line_chart('ventilation_rate')
    # draw_accumulative_chart('pm25_intake')
    print 'lalala~'
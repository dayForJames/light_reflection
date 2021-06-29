import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    y = x**2 + 5 * math.cos(x - 1)
    return y


def f_(x):
    y = 2 * x - 5 * math.sin(x - 1)
    return y


def calc():
    x_i = -10**4
    approximation = 10**-10
    for i in range(-10**4, 10**4):
        if abs(f(x_i) / f_(x_i)) < approximation:
            return x_i
        else:
            x_i -= f(x_i) / f_(x_i)
    return 'Sowwy'

'''
5x+12x-3x=23
-10x+8x-2x=10
25x+30x-7x=3
'''


def determ(mt):
    s = a = 1
    '''
    [5, 12, -3]
    [-10, 8, -2]
    [25, 30, -7]
    '''
    for i in range(3):
        s *= mt[i][i]
        a *= mt[i][2 - i]
    s -= a

    s += mt[0][2] * mt[1][0] * mt[2][1] + mt[2][0] * mt[0][1] * mt[1][2] - \
        (mt[0][0] * mt[2][1] * mt[1][2] + mt[2][2] * mt[0][1] * mt[1][0])

    return s


def matrix():
    n = 3
    ex = []
    mt = [[], [], []]

    for i in range(n):
        line = input()
        ex.append(line)

    cop = ex.copy()

    for i in range(n):
        for j in range(n):
            a = cop[i].find('x')
            if '+' in cop[i][:a]:
                mt[i].append(int(cop[i][1:a]))
            else:
                mt[i].append(int(cop[i][:a]))
            cop[i] = cop[i][a + 1:]

    vec = [int(cop[0][1:]), int(cop[1][1:]), int(cop[2][1:])]
    s = determ(mt)
    det_box = [s]

    '''
    [5, 12, -3]
    [-10, 8, -2]
    [25, 30, -7]
    '''

    if s != 0:
        for i in range(n):
            cop = mt.copy()

            for j in range(n):
                cop[j][i] = vec[j]

            det_box.append(determ(cop))

        print(f'Answer is ', end="")

        for i in range(n):
            print(det_box[i + 1] / det_box[0], ',', end="")
    else:
        print('Equation has no roots')


def vec_length(v):
    return ((v[1][0] - v[0][0])**2 + (v[1][1] - v[0][1])**2)**0.5


def vec_angle(v_1, v_2):
    return ((v_1[1][0] - v_1[0][0]) * (v_2[1][0] - v_2[0][0]) +
            (v_1[1][1] - v_1[0][1]) * (v_2[1][1] - v_2[0][1])) / (vec_length(v_1) * vec_length(v_2))


def vector_reflection(main_vec, vec):
    #   главный вектор
    main_l = vec_length(main_vec)

    # угол между ними
    cos_a = vec_angle(main_vec, vec)
    print(cos_a)

    x_o, y_o = touch_points(main_vec, vec)

    l = ((x_o - main_vec[0][0])**2 + (y_o - main_vec[0][1])**2)**0.5
    x_1 = x_o

    y_1 = l + y_o if cos_a == (x_o * x_1 + y_o * (l + y_o)) / l else y_o - l


    '''m = x_o - main_vec[0][0]
    n = y_o - main_vec[0][1]

    c = np.cos(2 * np.arccos(cos_a)) * l_ao**2

    q, e, v = n**2 + 1, c * n / m**2, (c / m)**2 - l_ao**2

    if e**2 - q * v < 0:
        x_l, y_l = main_vec[0][0], main_vec[0][1]
    else:
        y_l1, y_l2 = (e - (e**2 - q * v)**0.5) / q + y_o, (e + (e**2 - q * v)**0.5) / q + y_o
        x_l1, x_l2 = (c - n * (y_l1 - y_o)) / m + x_o, (c - n * (y_l2 - y_o)) / m + x_o

        if (x_l1 - x_o)**2 + (y_l1 - y_o)**2 == l_ao**2:
            x_l, y_l = x_l1, y_l1
        else:
            x_l, y_l = x_l2, y_l2'''

    fig, ax = plt.subplots()

    ax.plot([vec[0][0], vec[1][0]], [vec[0][1], vec[1][1]])
    ax.plot([main_vec[0][0], x_o, x_1], [main_vec[0][1], y_o, y_1])
    fig.savefig('Vector of the sun light')


def add_vector_to_show(ax, x, y):
    ax.plot(x, y)


def shortest_way(m, v):
    box_l = []
    for i in v:
        x_o, y_o = touch_points(m, i)
        if x_o is not None:
            ln = 1
            box_l.append(ln)
        else:
            box_l.append(None)
    ind = box_l.index(min(box_l))

    return ind


def touch_points(v_1, v_2):
    k_1 = (v_1[1][1] - v_1[0][1]) / (v_1[1][0] - v_1[0][0])
    k_2 = (v_2[1][1] - v_2[0][1]) / (v_2[1][0] - v_2[0][0])

    if k_1 != k_2:
        b_1 = v_1[1][1] - v_1[1][0] * k_1
        b_2 = v_2[1][1] - v_2[1][0] * k_2

        x_o = (b_2 - b_1) / (k_1 - k_2)
        y_o = k_1 * x_o + b_1

        return x_o, y_o
    return None, None

def main():
    #fig, ax = plt.subplots()

    main_vec = ((1, 2), (3, 3))
    box_vec = [((5, 6), (8, 3))]
    #, ((4, 2), (7, 1))

    for i in box_vec:
        #ax.plot([i[0][0], i[1][0]], [i[0][1], i[1][1]], color='red')
        pass

    while True:
        #print(shortest_way(main_vec, box_vec))
        ind = shortest_way(main_vec, box_vec)

        vector_reflection(main_vec, box_vec[ind])
        break

    #plt.show()

if __name__ == '__main__':
    main()

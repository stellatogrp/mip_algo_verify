import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def main():
    two_step_times = [0.0008041858673095703, 0.006671905517578125, 0.06252503395080566, 0.15401005744934082, 0.1582329273223877, 0.38556909561157227, 0.4091348648071289, 0.33592915534973145, 0.3364579677581787, 0.5243310928344727, 1.08536696434021, 1.5955560207366943, 2.256135940551758, 6.065163850784302, 5.48886513710022, 10.114967107772827, 12.339134216308594, 17.10669994354248, 49.48233604431152, 64.27934885025024, 90.23336791992188, 126.24582600593567, 146.57884001731873, 329.1829788684845, 369.25311398506165, 805.152822971344]
    one_step_times = [0.002045869827270508, 0.01322317123413086, 0.05414104461669922, 0.13153910636901855, 0.19913601875305176, 0.37172484397888184, 0.4937160015106201, 0.5264110565185547, 0.346235990524292, 0.690140962600708, 1.436068058013916, 1.745574951171875, 2.4932239055633545, 5.049808025360107, 5.695472002029419, 8.283781051635742, 9.491904973983765, 15.08427882194519, 27.69964909553528, 60.59195804595947, 110.91231203079224, 115.43468809127808, 281.5427448749542, 414.6541678905487, 3251.2080438137054, 711.2473139762878]
    one_step_callback_times = [0.0016279220581054688, 0.020753860473632812, 3.3960089683532715, 2.2347798347473145, 4.823691129684448, 5.009075164794922, 4.395067930221558, 1.3788988590240479, 1.855638027191162, 3.870032787322998, 36.839226722717285, 34.44493508338928, 64.9094979763031, 57.621675968170166, 73.2097761631012, 160.28383207321167, 144.23790383338928, 73.18647408485413, 274.84813499450684, 95.40954184532166, 135.84767317771912, 225.53333687782288, 289.12672305107117, 410.8095860481262, 762.5090072154999, 1053.7929899692535]

    fig, ax = plt.subplots()
    ax.plot(range(1, len(two_step_times)+1), two_step_times, label='two step times')
    ax.plot(range(1, len(one_step_times)+1), one_step_times, label='one step times')
    ax.plot(range(1, len(one_step_callback_times)+1), one_step_callback_times, label='one step + callback times')

    ax.set_xlabel(r'$K$')
    ax.set_ylabel('Solvetime(s)')
    ax.set_yscale('log')
    ax.set_title(r'NNQP VP, $n=20$')

    ax.legend()

    plt.show()


def main_bounds():
    bounds = [45.0, 16.845927224618816, 11.805483789894346, 7.018711511686336, 4.847662234904601, 3.1795344421073133, 2.500129389125241, 2.02802195432378, 1.676974758888299, 1.3781176804199458, 1.1352864389990738, 0.9337618555589021, 0.7682947734009443, 0.6319350977891934, 0.5197661436364731, 0.4274856024183188, 0.35156952258226454, 0.28913600040371396, 0.23778160767980216, 0.19554982887903144, 0.1608160241679159, 0.13225225147882883, 0.10876112440926916, 0.08944272218563862, 0.0735554839124717, 0.06049051685716044]
    fig, ax = plt.subplots()
    ax.plot(range(1, len(bounds) +1), bounds)
    ax.set_yscale('log')

    plt.show()


if __name__ == '__main__':
    # main()
    main_bounds()

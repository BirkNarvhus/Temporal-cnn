import matplotlib.pyplot as plt


class CsvReader:
    def __init__(self, filename, root):
        self.file = open(root + filename, "r")

    def __del__(self):
        self.file.close()

    def remove_tensor(self, line):
        return line[7:-1]

    def read_lines(self):
        out = {"params": 0}
        curr_epoch = 1

        for line in self.file.readlines():
            line = line.strip()
            splits = line.split(',')

            if len(splits) == 0:
                continue

            if curr_epoch not in out and len(splits) != 1:
                out[curr_epoch] = {}

            if splits[0] == "TEST":
                out[curr_epoch]["test"] = {"avg_loss": float(splits[1]) * 100,
                                           "acc": round(float(self.remove_tensor(splits[-1])), 1)}
                curr_epoch += 1
            elif len(splits) == 1:
                if splits[0] != "":
                    out["params"] = int(splits[0])
            else:
                if "training" not in out[curr_epoch]:
                    out[curr_epoch]["training"] = []

                out[curr_epoch]["training"].append([*splits[1:]])

        return out


resroot = "../result/lstmattn/"
filename = "LSTMATTN-hs64-l2-d0-P-21_11_2023_17_30_58.csv"

name = filename.split('-')
name = '-'.join(name[:5])

csvreader = CsvReader(filename, resroot)

data = csvreader.read_lines()


cur_epc = 1
x = []
y = []
while cur_epc in data:
    training_data = data[cur_epc]["training"]

    for idx, (step, data_trained, loss) in enumerate(training_data):
        x.append(int(step))
        y.append(float(loss))

    plt.plot([x[-1], x[-1]], [y[-1] - 0.3, y[-1] + 1], color="red")
    if "test" in data[cur_epc]:
        plt.text(x[-1] - 20000, y[-1] + 1, str(data[cur_epc]["test"]["acc"]) + "%", fontsize=8)
    cur_epc += 1
plt.title(name)
plt.plot(x, y)
# plot the sample
plt.show()



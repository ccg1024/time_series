import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class LSTM(nn.Module):
    def __init__(self, n_hidden=51) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(input_size=1, hidden_size=n_hidden, num_layers=2)
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x, future=0):
        outputs = []
        n_samples = x.size(0)

        h_t = torch.zeros(2, n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(2, n_samples, self.n_hidden, dtype=torch.float32)

        for input_t in x.split(1, dim=1):
            _, (h_t, c_t) = self.lstm(input_t.unsqueeze(0).to(torch.float32), (h_t, c_t))
            output = self.fc(h_t[-1]).squeeze(0)
            outputs.append(output)

        # predict
        for i in range(future):
            _, (h_t, c_t) = self.lstm(outputs[-1].unsqueeze(0), (h_t, c_t))
            output = self.fc(h_t[-1]).squeeze(0)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs


if __name__ == "__main__":
    # data shape: (100, 1000) means 100 samples, 1000 features
    datas = torch.load('datas/traindata.pt')
    train_input = torch.from_numpy(datas[3:, :-1])
    train_target = torch.from_numpy(datas[3:, 1:])

    test_input = torch.from_numpy(datas[:3, :-1])
    test_target = torch.from_numpy(datas[:3, 1:])

    model = LSTM()
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.8)

    n_step = 10
    for i in range(n_step):
        print('Step ', i)


        def closure():
            optimizer.zero_grad()
            out = model(train_input)  # 97, 999
            loss = criterion(out, train_target)
            print("loss", loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            future = 1000
            pred = model(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print("test loss", loss.item())
            y = pred.detach().numpy()

        plt.figure(figsize=(12, 6))
        plt.title(f"Step {i + 1}")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        n = train_input.shape[1]  # 999


        def draw(y_i, color):
            plt.plot(np.arange(n), y_i[:n], color, linewidth=2.0)
            plt.plot(np.arange(n, n + future), y_i[n:], color + ":", linewidth=2.0)


        draw(y[0], 'b')
        plt.savefig('datas/predict_single%d.png' % i)
        plt.close()

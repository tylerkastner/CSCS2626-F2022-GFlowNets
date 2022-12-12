import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

y = [.14088981, 0.24406315, 0.36481625, 1.3355317]
x = [1,2,3,4]

data = list(zip(x,y))

print(data)

df = pd.DataFrame(data, columns = ['Grid dimension','Weighted total variation distance'])

print(df)

sns.lineplot(data=df, x='Grid dimension', y='Weighted total variation distance')

plt.xticks([1,2,3,4])

# plt.title('Reward error')

plt.show()


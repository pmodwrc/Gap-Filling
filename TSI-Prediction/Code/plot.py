import matplotlib.pyplot as plt

t = [1,2,4,8,16,24]
train_loss = [0.0018, 0.0013, 0.0012, 0.0007, 0.0006, 0.0002]
test_loss = [0.0161,0.0128,0.0135,0.0172,0.0101, 0.0171]

plt.plot(t, test_loss, label = 'Test Loss')
plt.plot(t, train_loss, label = 'Train Loss')
plt.xlabel('Months')
plt.ylabel('Loss')
plt.legend()
plt.show()
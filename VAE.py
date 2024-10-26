import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(VAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim * 2)  # 产生均值和方差
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 假设数据在[0, 1]之间
        )

    def encode(self, x):
        h = self.encoder(x)
        mean, log_var = torch.chunk(h, 2, dim=1)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_var = self.encode(x.view(-1, input_dim))
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var

# 超参数
input_dim = 784  # 例如，28x28的图像
hidden_dim = 400
z_dim = 20
learning_rate = 1e-3
batch_size = 128
epochs = 10

# 实例化模型
vae = VAE(input_dim, hidden_dim, z_dim)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# 加载数据集
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 训练模型
vae.train()
for epoch in range(epochs):
    for data in train_loader:
        x = data[0].view(-1, input_dim)
        optimizer.zero_grad()
        reconstructed_x, mean, log_var = vae(x)
        loss = criterion(reconstructed_x, x) + 0.5 * torch.sum(log_var - mean.pow(2) - log_var.exp())
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# 测试模型
vae.eval()
with torch.no_grad():
    x_test = next(iter(train_loader))[0].view(-1, input_dim)
    reconstructed_x, _, _ = vae(x_test)
    print('reconstructed_x.shape:', reconstructed_x.shape)
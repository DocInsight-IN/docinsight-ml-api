import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, activation='relu', epochs=200, batch_size=128):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.activation = nn.ReLU() if activation == 'relu' else nn.Sigmoid()
        self.epochs = epochs
        self.batch_size = batch_size

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            self.activation
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            self.activation
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def fit(self, X):
        X_train, X_test = train_test_split(X)
        train_loader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(X_train)), batch_size=self.batch_Size, shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.Tensor(X_test), torch.Tensor(X_test)), batch_size=self.batch_size)

        for epoch in range(self.epochs):
            train_loss = 0.0
            for data in train_loader:
                inputs, _ = data
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()
                train_loss = loss.item() * inputs.size(0)
            train_loss /= len(train_loader.dataset)

            test_loss = 0.0
            with torch.no_grad():
                for data in test_loader:
                    inputs, _ = data
                    outputs = self(inputs)
                    loss = self.criterion(outputs, inputs)
                    test_loss += loss.item() * inputs.size(0)
                test_loss /= len(test_loader.dataset)

            print(f'Epoch [{epoch + 1}/{self.epochs}], Train Loss : {train_loss:.4f}, Test Loss : {test_loss:.4f}')
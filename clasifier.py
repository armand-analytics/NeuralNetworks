import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Definir el modelo de red neuronal
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        # Aplanar la imagen de entrada de 28x28 a un vector de 784 elementos
        self.flatit = nn.Flatten()
        
        # Definir las capas de la red neuronal
        self.net = nn.Sequential(
            nn.Linear(28*28, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.Softmax(dim=1)  # Softmax en la dimensión de las clases
        )

    def forward(self, x):
        # Procesar la entrada a través de la red
        x = self.flatit(x)
        logits = self.net(x)
        return logits

# Crear una instancia de la red neuronal
model = NeuralNet()
print(model)

# Calcular el número de parámetros en la red neuronal
t = sum(p.numel() for p in model.parameters())
print(f'The number of parameters in the neural network architecture is {t}')

# Hiperparámetros
learning_rate = 0.1
epochs = 10
batch_size = 2000

# Cargar datos y definir transformaciones
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='datos', train=True, download=True, transform=transform)
val_dataset = datasets.MNIST(root='datos', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Configurar dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Definir la función de pérdida y el optimizador
fn_loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Función de entrenamiento
def train_loop(dataloader, model, fn_loss, optimizer):
    train_size = len(dataloader.dataset)
    nbatches = len(dataloader)
    model.train()
    
    loss_accum, accuracy_accum = 0, 0
    
    for nbatch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = fn_loss(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_accum += loss.item()
        accuracy_accum += (logits.argmax(1) == y).type(torch.float).sum().item()
        
        if nbatch % 10 == 0:
            ndata = nbatch * batch_size
            print(f'\tloss: {loss.item():>7f} [{ndata:>5d}/{train_size:>5d}]')

    loss_accum /= nbatches
    accuracy_accum /= train_size
    print(f'\tAccuracy: {(100 * accuracy_accum):>0.1f}% | Loss: {loss_accum:>8f}')

# Función de validación
def val_loop(dataloader, model, fn_loss):
    val_size = len(dataloader.dataset)
    nbatches = len(dataloader)
    model.eval()
    
    loss_val, accuracy = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss_val += fn_loss(logits, y).item()
            accuracy += (logits.argmax(1) == y).type(torch.float).sum().item()

    loss_val /= nbatches
    accuracy /= val_size
    print(f'\t\tAccuracy: {(100 * accuracy):>0.1f}% | Loss: {loss_val:>8f}')

# Entrenamiento y validación del modelo
for epoch in range(epochs):
    print(f'Epoch {epoch+1}\n-------------------------------')
    train_loop(train_loader, model, fn_loss, optimizer)
    val_loop(val_loader, model, fn_loss)
print('Done!')

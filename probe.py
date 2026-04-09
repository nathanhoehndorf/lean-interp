import torch
import torch.nn as nn
import torch.optim as optim

# Assume 'activations' is a list of [768] vectors we collected
# and 'labels' is a list of integers (number of variables)
# For this example, let's use dummy data to show the structure:
X = torch.randn(100, 768)  # 100 proof states, layer 6 activations
y = torch.randint(1, 5, (100, 1)).float()  # Random variable counts 1-4

# The "Probe" is just a single linear layer
# If the model 'knows' the count, this will converge quickly
probe = nn.Linear(768, 1) 
optimizer = optim.Adam(probe.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Training the probe...")
for epoch in range(100):
    optimizer.zero_grad()
    predictions = probe(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
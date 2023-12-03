import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class CustomNet(nn.Module):
    def __init__(self, batch_size, input_dim, output_dim):
        # batch_size is dim_Ay, input_dim is dim_Ax, output_dim is ncoil
        super(CustomNet, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def train_model(dim_Ax, dim_Ay, ii, ncoil=4, num_epochs=1000, lr=0.001):
    # Create the model, loss function, and optimizer
    model = CustomNet(dim_Ay, dim_Ax, ncoil)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    num_epochs = 1000
    datasets = load_datasets() # load A, C for each set
    for epoch in range(num_epochs):
        # Iterate over multiple sets of A and C
        for dataset in datasets:
            A, C = dataset['A'], dataset['C'] # A is torch(12, 100), C is torch(4, 100)

            # Forward pass
            B = model(A)

            # Compute the loss
            loss = criterion(torch.matmul(B, A) - C, torch.zeros_like(C))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), f'trained_model_dim_{ii}.pth')

if __name__ == '__main__':
    dim_Ax, dim_Ay = 12, 100
    ii = 1
    train_model(dim_Ax, dim_Ay, ii)
    model = torch.load(f'trained_model_dim_{ii}.pth')

    # After training, you can use the model to make predictions on new data
    # For example:
    test_A = torch.rand(dim_Ax, dim_Ay)
    predicted_B = model(test_A)

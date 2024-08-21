# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()
#     output, latent = model(x_train)
#     loss = criterion(output, x_train)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np


class BaseTrainer:
    def __init__(self, model, optimiser = "adam"):
        self.model = model

        if optimiser == "adam":
            self.optimiser = optim.Adam(
                self.model.parameters(), lr=0.0005,
            )
        else:
            self.optimiser = optim.SGD(self.model.parameters(), lr=0.0005, momentum=0.9)

    def _train(
        self,
        train_loader,
        loss_function=torch.nn.MSELoss(),
        num_epochs=50,
        batch_size=50,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        model = self.model.to(device)
        model.train()

        optimiser = self.optimiser

        for epoch in range(num_epochs):
            loss_history = []
            print(f"Epoch: {epoch}/{num_epochs-1}")
            print("-----------------------------")
            start = time.perf_counter()
            for batch_id, (data, label) in enumerate(DataLoader(train_loader, batch_size=batch_size, shuffle=True)):
                data = Variable(data.to(device))
                target = Variable(label.to(device))

                optimiser.zero_grad()
                preds, _ = model(data)
                # print(data.shape, preds.shape)
                loss = loss_function(preds, data)
                loss.backward()
                loss_history.append(loss.data.item())
                optimiser.step()

            print(f"Loss avg: {sum(loss_history)/len(loss_history)}")

            print(f"Time : {time.perf_counter() - start}")
            print("=============================")

        self.model = model
        return model
    def extract_latent(
        self,
        train_loader,
        batch_size = 50,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        model = self.model.to(device)
        model.eval()
        
        start = time.perf_counter()
        
        latent_array = []
        target_array = []
        
        for data, label in DataLoader(train_loader, batch_size=batch_size, shuffle=False):
            data = Variable(data.to(device))
            target = Variable(label.to(device))
            
            preds, latent = model(data)
            latent_array.append(latent.detach().cpu().numpy())
            target_array.append(target.detach().cpu().numpy())

        
        latent_np = np.concatenate(latent_array, axis = 0)
        target_np = np.concatenate(target_array, axis = 0)


        print(f"Time : {time.perf_counter() - start}")
        print("=============================")

        return latent_np, target_np
        
    def save_model(
        self,
        file_path: str,
        model,
    ) -> None:
        """Save the model

        Args:
            file_path (str): Path dest
            model (nn, optional): Model. Defaults to None.
        """
        torch.save(model.state_dict(), file_path)
        
class AAMTrainer:
    def __init__(self, model, optimiser = "adam"):
        self.model = model

        if optimiser == "adam":
            self.optimiser = optim.Adam(
                self.model.parameters(), lr=0.0005,
            )
        else:
            self.optimiser = optim.SGD(self.model.parameters(), lr=0.0005, momentum=0.9)

    def _train(
        self,
        train_loader,
        loss_function=torch.nn.MSELoss(),
        num_epochs=50,
        batch_size=50,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        model = self.model.to(device)
        model.train()

        optimiser = self.optimiser

        for epoch in range(num_epochs):
            loss_history = []
            print(f"Epoch: {epoch}/{num_epochs-1}")
            print("-----------------------------")
            start = time.perf_counter()
            for batch_id, (data, label) in enumerate(DataLoader(train_loader, batch_size=batch_size, shuffle=True)):
                data = Variable(data.to(device))
                target = Variable(label.to(device))

                optimiser.zero_grad()
                # print(target)
                preds, _ = model(data, target)
                # print(data.shape, preds.shape)
                loss = loss_function(preds, data)
                loss.backward()
                loss_history.append(loss.data.item())
                optimiser.step()

            print(f"Loss avg: {sum(loss_history)/len(loss_history)}")

            print(f"Time : {time.perf_counter() - start}")
            print("=============================")

        self.model = model
        return model
    def extract_latent(
        self,
        train_loader,
        batch_size = 50,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        model = self.model.to(device)
        model.eval()
        
        start = time.perf_counter()
        
        latent_array = []
        target_array = []
        
        for data, label in DataLoader(train_loader, batch_size=batch_size, shuffle=False):
            data = Variable(data.to(device))
            target = Variable(label.to(device))
            
            preds, latent = model(data)
            latent_array.append(latent.detach().cpu().numpy())
            target_array.append(target.detach().cpu().numpy())

        
        latent_np = np.concatenate(latent_array, axis = 0)
        target_np = np.concatenate(target_array, axis = 0)


        print(f"Time : {time.perf_counter() - start}")
        print("=============================")

        return latent_np, target_np
        
    def save_model(
        self,
        file_path: str,
        model,
    ) -> None:
        """Save the model

        Args:
            file_path (str): Path dest
            model (nn, optional): Model. Defaults to None.
        """
        torch.save(model.state_dict(), file_path)

class ClassicTrainer:
    def __init__(self, model, optimiser = "adam"):
        self.model = model

        if optimiser == "adam":
            self.optimiser = optim.Adam(
                self.model.parameters(), lr=0.0005,
            )
        else:
            self.optimiser = optim.SGD(self.model.parameters(), lr=0.0005, momentum=0.9)

    def _train(
        self,
        train_loader,
        loss_function=torch.nn.MSELoss(),
        num_epochs=50,
        batch_size=50,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        model = self.model.to(device)
        model.train()

        optimiser = self.optimiser

        for epoch in range(num_epochs):
            loss_history = []
            print(f"Epoch: {epoch}/{num_epochs-1}")
            print("-----------------------------")
            start = time.perf_counter()
            for batch_id, (data, label) in enumerate(DataLoader(train_loader, batch_size=batch_size, shuffle=True)):
                data = Variable(data.to(device))
                target = Variable(label.type(torch.LongTensor).to(device))

                optimiser.zero_grad()
                preds = model(data)
                # print(target.shape, preds.shape)
                # print(target)
                loss = loss_function(preds, target)
                loss.backward()
                loss_history.append(loss.data.item())
                optimiser.step()

            print(f"Loss avg: {sum(loss_history)/len(loss_history)}")

            print(f"Time : {time.perf_counter() - start}")
            print("=============================")

        self.model = model
        return model
    def _test(
        self,
        test_loader,
        loss_function=torch.nn.MSELoss(),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        y_dim = 4,
        batch_size = 50
    ) -> None:
        model = self.model.to(device)
        model.eval()
        confusion_matrix = np.zeros((y_dim, y_dim))

        test_loss = 0
        correct = 0

        start = time.perf_counter()
        for batch_id, (data, label) in enumerate(DataLoader(test_loader, batch_size=batch_size, shuffle=True)):
            # print("Batch_id: ", batch_id)
            data = Variable(data.to(device))
            target = Variable(label.type(torch.LongTensor).to(device))

            # print(data.shape)
            output = model(data)
            test_loss += loss_function(output, target).data.item()
            # print(output)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            # print(type(pred), type(target))
            pred_ = pred.cpu().numpy()
            target_ = target.cpu().numpy()
            
            for i in range(pred_.shape[0]):
                confusion_matrix[pred_[i], target_[i]] += 1

        test_loss = test_loss
        test_loss /= len(test_loader)  # loss function already averages over batch size
        accuracy = 100.0 * correct / len(test_loader)
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(test_loader), accuracy
            )
        )
        print(f"Time : {time.perf_counter() - start}")
        
        return confusion_matrix
    
class TwoStepTrainer:
    def __init__(self, model, optimiser = "adam"):
        self.model = model

        if optimiser == "adam":
            self.optimiser = optim.Adam(
                self.model.parameters(), lr=0.0005,
            )
        else:
            self.optimiser = optim.SGD(self.model.parameters(), lr=0.0005, momentum=0.9)

    def _train(
        self,
        train_loader,
        loss_function=torch.nn.MSELoss(),
        num_epochs=50,
        batch_size=50,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        model = self.model.to(device)
        model.train()

        optimiser = self.optimiser

        for epoch in range(num_epochs):
            loss_history = []
            print(f"Epoch: {epoch}/{num_epochs-1}")
            print("-----------------------------")
            start = time.perf_counter()
            for batch_id, (data, label) in enumerate(DataLoader(train_loader, batch_size=batch_size, shuffle=True)):
                data = Variable(data.to(device))
                target = Variable(label.to(device))

                optimiser.zero_grad()
                preds, _ = model(data)
                # print(data.shape, preds.shape)
                loss = loss_function(preds, data)
                loss.backward()
                loss_history.append(loss.data.item())
                optimiser.step()

            print(f"Loss avg: {sum(loss_history)/len(loss_history)}")

            print(f"Time : {time.perf_counter() - start}")
            print("=============================")

        self.model = model
        return model
    def _train2(
        self,
        train_loader,
        loss_function=torch.nn.MSELoss(),
        num_epochs=50,
        batch_size=50,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        model = self.model.to(device)
        model.train()

        optimiser = self.optimiser

        for epoch in range(num_epochs):
            loss_history = []
            print(f"Epoch: {epoch}/{num_epochs-1}")
            print("-----------------------------")
            start = time.perf_counter()
            for batch_id, (data, label) in enumerate(DataLoader(train_loader, batch_size=batch_size, shuffle=True)):
                data = Variable(data.to(device))
                target = Variable(label.to(device))

                optimiser.zero_grad()
                latent = model(data)
                # print(data.shape, preds.shape)
                loss = loss_function(latent, target)
                loss.backward()
                loss_history.append(loss.data.item())
                optimiser.step()

            print(f"Loss avg: {sum(loss_history)/len(loss_history)}")

            print(f"Time : {time.perf_counter() - start}")
            print("=============================")

        self.model = model
        return model
    def extract_latent(
        self,
        train_loader,
        batch_size = 50,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        model = self.model.to(device)
        model.eval()
        
        start = time.perf_counter()
        
        latent_array = []
        target_array = []
        
        for data, label in DataLoader(train_loader, batch_size=batch_size, shuffle=False):
            data = Variable(data.to(device))
            target = Variable(label.to(device))
            
            try:
                preds, latent = model(data)
            except ValueError:
                latent = model(data)
                
            latent_array.append(latent.detach().cpu().numpy())
            target_array.append(target.detach().cpu().numpy())

        
        latent_np = np.concatenate(latent_array, axis = 0)
        target_np = np.concatenate(target_array, axis = 0)


        print(f"Time : {time.perf_counter() - start}")
        print("=============================")

        return latent_np, target_np
        
    def save_model(
        self,
        file_path: str,
        model,
    ) -> None:
        """Save the model

        Args:
            file_path (str): Path dest
            model (nn, optional): Model. Defaults to None.
        """
        torch.save(model.state_dict(), file_path)
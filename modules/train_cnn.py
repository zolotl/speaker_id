import numpy as np
from torch import nn, optim
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from datasets import train_data, train_dataloader, val_dataloader, speaker_label_map
from model import simpleDCNN


num_classes = len(np.unique(list(train_data.keys())))
model = simpleDCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_dataloader)}")



def get_true_and_predicted_labels_cnn(model, dataloader, speaker_label_map):
  model.eval()
  true_labels = []
  predicted_labels = []

  with torch.no_grad():
    for inputs, labels in dataloader:
      outputs = model(inputs)
      _, predictions = torch.max(outputs, 1)
      true_labels.extend(labels.numpy())
      predicted_labels.extend(predictions.numpy())

  # Invert the speaker_to_label dictionary for mapping back the labels to speaker names
  label_to_speaker = {v: k for k, v in speaker_label_map.items()}

  # Map back the numerical labels to speaker names
  true_speaker_labels = [label_to_speaker[l] for l in true_labels]
  predicted_speaker_labels = [label_to_speaker[l] for l in predicted_labels]

  return true_speaker_labels, predicted_speaker_labels

true_speaker_labels, predicted_speaker_labels = get_true_and_predicted_labels_cnn(model, val_dataloader, speaker_label_map)

accuracy = accuracy_score(true_speaker_labels, predicted_speaker_labels)
confusion_mat = confusion_matrix(true_speaker_labels, predicted_speaker_labels)
report = classification_report(true_speaker_labels, predicted_speaker_labels)

print("Accuracy:", accuracy)
print("Confusion matrix:\n", confusion_mat)
print("Classification report:\n", report)

torch.save(model.state_dict(), "../output/models/model_1_{}".format(accuracy))
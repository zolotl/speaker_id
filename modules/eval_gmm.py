from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from train_gmm import gmms

from preprocessing import load_fsdd_data

val_directory = "../datasets/processed/val"
val_data = load_fsdd_data(val_directory, max_length=150)

# Evaluate GMMs on test data
def get_true_and_predicted_labels_gmm(gmms, test_data):
  true_labels = []
  predicted_labels = []

  for speaker_id, samples in test_data.items():
    for mfccs, _ in samples:  # Ignore the digit label
      true_labels.append(speaker_id)

      scores = {sid: gmm.score(scaler.transform(mfccs) if scaler else mfccs) for sid, (gmm, scaler) in gmms.items()}
      predicted_speaker_id = max(scores, key=scores.get)
      predicted_labels.append(predicted_speaker_id)

  return true_labels, predicted_labels

true_labels, predicted_labels = get_true_and_predicted_labels_gmm(gmms, val_data)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")

# Confusion matrix
confusion = confusion_matrix(true_labels, predicted_labels)
print("Confusion matrix:")
print(confusion)

report = classification_report(true_labels, predicted_labels)
print(report)
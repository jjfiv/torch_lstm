# torch_lstm
Package containing LSTM-based classifiers in Torch.

## What's here now?

Inside ``torch_lstm.classifier`` there's a ``SequentialClassifier`` with a clean interface that is highly-customizable.

```python
  y_train = [1, 1, 0, 0]
  X_train = ["I am happy.", "This is great!", "I am sad.", "This is bad."]
  # We need to learn the vocab sizes of the training data in order to construct our classifier.
  config = DatasetConfig()
  X_ready = config.fit_transform(X_train)

  # Create our classifier.
  clf = SequenceClassifier(
      # first argument is the updated DatasetConfig.
      config,
      # character-embeddings useful if you have messy data, like, Twitter, or you're doing something where the shape of words matter.
      # Slow if you don't need it.
      char_dim=0,
      char_lstm_dim=0,
      # If you add pre-trained embeddings to your DataSetConfig, this is ignored.
      word_dim: int = 300,
      # We use a bidirectional LSTM, this is the width of each half.
      lstm_size=100,
      # This is a layer directly atop the embeddings (if you're using pretrained) to recontextualize them for your task.
      gen_layer=100,
      # This configures the MLP after the LSTM.
      hidden_layer=100,
      # Output labels (how many classes?)
      labels=[0, 1],
      # Dropout is really slow if you don't have tons of data, you might not need it.
      dropout=0.0,
      # This is the nonlinearity that BERT uses, apparently. No effect on poetry dataset.
      activation='gelu',
      # To make your classifier scale to longer texts, you can average sliding windows of (width, step) embeddings.
      averaging=(6,4)
  )
  # Training:
  optimizer = torch.optim.Adam(params=clf.parameters())
  loss_function = torch.nn.CrossEntropyLoss()
  train_epoch(clf, optimizer, loss_function, X_ready, y_train)
```

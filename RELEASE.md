# 0.8.1

Changed setup.py to allow this package to be installed in AWS deep learning AMIs where "tensorflow" package is named
"tensorflow-gpu".

Fixed bug whereby pruning would not be done if keras was imported via `import keras`. Using
`import tensorflow.keras as keras` or `from tensorflow.python import keras` were working fine though.
Added unit tests that import keras in various different ways to ensure this package works regardless.

# 0.8.0

Initial functional package. Tested via unit tests and via integration into an unrelated image classification pipeline
based on MobilenetV2.

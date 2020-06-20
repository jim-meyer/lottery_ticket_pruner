import unittest

# We just make sure that we're finding all the expected prunable layers since layer names are different
# depending on how we import keras. Some of the different ways that keras can be imported:
#   import tensorflow.keras as keras
#   from tensorflow.python import keras
#   import keras
try:
    import keras
    KERAS_IMPORTED = True
except ImportError:
    KERAS_IMPORTED = False

import lottery_ticket_pruner


@unittest.skipIf(not KERAS_IMPORTED, 'Skipping unit tests that uses `import keras` since keras is not installed per se')
class TestImportKeras(unittest.TestCase):
    def test_inception_v3(self):
        model = keras.applications.InceptionV3(input_shape=(299, 299, 3),
                                               weights='imagenet',
                                               include_top=True,
                                               pooling='max')
        pruner = lottery_ticket_pruner.LotteryTicketPruner(model)
        self.assertEqual(95, len(pruner.prune_masks_map))


if __name__ == '__main__':
    unittest.main()

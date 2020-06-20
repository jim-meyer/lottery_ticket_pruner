import unittest

# We just make sure that we're finding all the expected prunable layers since layer names are different
# depending on how we import keras. Some of the different ways that keras can be imported:
#   import tensorflow.keras as keras
#   from tensorflow.python import keras
#   import keras
from tensorflow.python import keras

import lottery_ticket_pruner


class TestFromTensorflowPythonImportKeras(unittest.TestCase):
    def test_inception_v3(self):
        if hasattr(keras.applications, 'InceptionV3'):
            factory_func = keras.applications.InceptionV3
        elif hasattr(keras.applications.inception_v3, 'InceptionV3'):
            factory_func = keras.applications.inception_v3.InceptionV3
        else:
            raise Exception('Cannot find InceptionV3 while using `from tensorflow.python import keras`')
        model = factory_func(input_shape=(299, 299, 3),
                             weights='imagenet',
                             include_top=True,
                             pooling='max')
        pruner = lottery_ticket_pruner.LotteryTicketPruner(model)
        self.assertEqual(95, len(pruner.prune_masks_map))


if __name__ == '__main__':
    unittest.main()

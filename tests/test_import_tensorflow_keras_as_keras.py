import unittest

# We just make sure that we're finding all the expected prunable layers since layer names are different
# depending on how we import keras. Some of the different ways that keras can be imported:
#   import tensorflow.keras as keras
#   from tensorflow.python import keras
#   import keras
import tensorflow.keras as keras

import lottery_ticket_pruner


class TestImportTensorflowKerasAsKeras(unittest.TestCase):
    def test_mobilenet_v2_from_tf2(self):
        model = keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               weights='imagenet',
                                               include_top=True,
                                               pooling='max')
        pruner = lottery_ticket_pruner.LotteryTicketPruner(model)
        self.assertEqual(53, len(pruner.prune_masks_map))


if __name__ == '__main__':
    unittest.main()

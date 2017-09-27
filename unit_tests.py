import unittest
import my_answers
import numpy as np

class TestWindowing(unittest.TestCase):

    def testSingleWindow(self):
        X, y = my_answers.window_transform_series(np.array([1,3,5]), 2)
        self.assertTrue((X == np.array([[1,3]])).all())
        self.assertTrue((y == np.array([5])).all())

    def testTwoWindows(self):
        X, y = my_answers.window_transform_series(np.array([1,3,5,9]), 2)
        self.assertTrue((X == np.array([[1,3], [3,5]])).all())
        self.assertTrue((y == np.array([[5], [9]])).all())

    def testOddWindows(self):
        X, y = my_answers.window_transform_series(np.array([1,3,5,7,9,11,13]), 2)
        self.assertTrue((X == np.array([[1,3], [3,5], [5,7], [7,9], [9,11]])).all())
        self.assertTrue((y == np.array([[5], [7], [9], [11], [13]])).all())


class TestWindowingText(unittest.TestCase):

    def testSingleWindow(self):
        inputs, outputs = my_answers.window_transform_text("The", 2, 1)
        self.assertEqual(inputs, ['Th'])
        self.assertEqual(outputs, ['e'])

    def testTwoWindows(self):
        inputs, outputs = my_answers.window_transform_text("Thes", 2, 1)
        self.assertEqual(inputs, ['Th', 'he'])
        self.assertEqual(outputs, ['e', 's'])

    def testStepSize2(self):
        inputs, outputs = my_answers.window_transform_text("These", 2, 2)
        self.assertEqual(inputs, ['Th', 'es'])
        self.assertEqual(outputs, ['e', 'e'])



if __name__ == "__main__":
    unittest.main()
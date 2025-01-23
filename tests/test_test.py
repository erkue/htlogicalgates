import unittest

print("Hi")

class TestCircuit(unittest.TestCase):
    def test_constructor(self):
        print("Ji")
        self.assertEqual(True, True)
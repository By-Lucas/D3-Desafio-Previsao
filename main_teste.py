import unittest
from .main import d3

class TestMain(unittest.TestCase):
    def test_candle_value_main(self):
        self.assertAlmostEqual(d3.main())

if __name__ == '__name__':
    unittest.main()
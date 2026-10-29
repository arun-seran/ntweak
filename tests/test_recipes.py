import unittest

from ntweak.recipes import normalize_recipe, recipe_id


class RecipeTests(unittest.TestCase):
    def test_neuronary_shape_and_sampling(self):
        result = normalize_recipe({"name": "Example", "neurons": [{"layer": 22, "neuron": 8, "dose": -0.7}],
                                   "sampling": {"temperature": 0, "repetition_penalty": 1.2}})
        self.assertEqual(result["neurons"][0], {"layer": 22, "neuron": 8, "dose": -0.7, "mode": "add", "enabled": True, "label": ""})
        self.assertEqual(result["sampling"]["temperature"], 0)
        self.assertEqual(result["sampling"]["repetition_penalty"], 1.2)

    def test_explorer_aliases_and_multiply(self):
        result = normalize_recipe({"steer_neurons": [{"layer": 2, "n": 9, "dose": 0}], "method": {"op": "mul"}})
        self.assertEqual(result["neurons"][0]["mode"], "mul")
        self.assertEqual(result["neurons"][0]["neuron"], 9)

    def test_reject_invalid_coordinates_and_nonfinite_doses(self):
        for row in [{"layer": -1, "neuron": 0}, {"layer": 0.5, "neuron": 0}, {"layer": True, "neuron": 0},
                    {"layer": 0, "neuron": 0, "dose": float("nan")}, {"layer": 0, "neuron": 0, "dose": float("inf")},
                    {"layer": 0, "neuron": 0, "dose": None}, {"layer": 0, "neuron": 0, "mode": "eval"}]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                normalize_recipe([row])

    def test_only_public_recipe_ids_are_used_for_requests(self):
        self.assertEqual(recipe_id("https://neuronary.org/r/9j43ped5"), "9j43ped5")
        self.assertEqual(recipe_id("9j43ped5"), "9j43ped5")
        for value in ["http://127.0.0.1/private", "https://neuronary.org.evil/r/abc", "../secrets", "https://neuronary.org/r/abc/def"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                recipe_id(value)

    def test_limits(self):
        with self.assertRaises(ValueError):
            normalize_recipe({"sampling": {"max_new_tokens": 999999}})
        with self.assertRaises(ValueError):
            normalize_recipe({"neurons": [{}] * 129})


if __name__ == "__main__":
    unittest.main()

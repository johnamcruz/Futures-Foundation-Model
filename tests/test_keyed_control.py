from futures_foundation.finetune.keyed_control import shuffle_training_keys


def test_keyed_control_defaults_to_complete_key_permutation():
    keys = [("a", 1), ("b", 2), ("c", 3)]
    assert shuffle_training_keys(object(), keys, [2, 0, 1]) == [keys[2], keys[0], keys[1]]


def test_keyed_control_allows_labeler_to_preserve_feature_identity():
    class Labeler:
        @staticmethod
        def shuffle_training_keys(keys, permutation):
            return [(key[0], keys[source][1]) for key, source in zip(keys, permutation)]

    keys = [("stream-a", "target-a"), ("stream-b", "target-b")]
    assert shuffle_training_keys(Labeler(), keys, [1, 0]) == [
        ("stream-a", "target-b"), ("stream-b", "target-a")]

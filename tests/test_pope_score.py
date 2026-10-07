from eval.pope_score import prediction


def test_pope_negative_word_rule():
    assert prediction("No, there is not.") == 0
    assert prediction("I don't see one.") == 0


def test_pope_positive_word_rule():
    assert prediction("Yes, there is a dog.") == 1

def semantic_tests(word_vectors):
    print("\nSemantic Tests!\n")

    most_similar = word_vectors.most_similar(
        positive=['woman', 'king'], negative=['man'], topn=3
    )
    # woman + king - man = ?
    print(
        "Man is to king as woman is to {}. (Top 3)".format(
            ', '.join(map(lambda t: t[0], most_similar))
        )
    )

    print(
        "From breakfast, cereal, dinner and lunch -- {} does not match.".format(
            word_vectors.doesnt_match("breakfast cereal dinner lunch".split())
        )
    )

    print(
        "Similarity!\nman -- woman: {}\nman -- silver: {}".format(
            word_vectors.similarity('man', 'woman'),
            word_vectors.similarity('man', 'silver')
        )
    )

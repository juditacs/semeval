def twitter_candidates(word, dictionary, stat):
    hashtag = '#' + word
    candidates = set()
    if hashtag in dictionary:
        stat['hashtag'].add(word)
        candidates.add(hashtag)
    candidates |= set(norvig_spellchecker(word, dictionary))
    if candidates:
        stat['norvig'].add(u'{0}\t{1}'.format(word, '\t'.join(candidates)))
    nodup = trim_dup_letters(word)
    if nodup in dictionary:
        stat['dupl'].add(u'{0}\t{1}'.format(word, nodup))
        candidates.add(nodup)
    parts = part_of_vocab(word, dictionary)
    if parts:
        stat['parts'].add('{0}\t{1}'.format(word, '\t'.join('{0} {1}'.format(p[0], p[1]) for p in parts)))
    for a, b in parts:
        candidates.add(a)
        candidates.add(b)
    return candidates


def trim_dup_letters(word):
    new_w = word[0]
    for c in word:
        if not new_w[-1] == c:
            new_w += c
    return new_w


def norvig_spellchecker(word, dictionary, dist=2):
    candidates = norvig_candidates(word, dist)
    return filter(lambda x: x in dictionary, candidates)


def norvig_candidates(word, dist=2):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    return set(deletes + transposes + inserts + replaces)


def part_of_vocab(word, dictionary):
    if len(word) < 5:
        return []
    splits = [(word[:i], word[i:]) for i in range(3, len(word) - 2)]
    parts = []
    for a, b in splits:
        if a in dictionary and b in dictionary:
            parts.append((a, b))
    return parts

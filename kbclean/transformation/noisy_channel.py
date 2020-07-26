from difflib import SequenceMatcher
from collections import Counter


class NoisyChannel:
    def __init__(self, string_pairs):
        self.transform_dist = self.transformation_distribution(string_pairs)

    def longest_common_substring(self, str1, str2):
        seqMatch = SequenceMatcher(None, str1, str2)

        match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))

        if match.size != 0:
            return str1[match.a : match.a + match.size]
        else:
            return ""

    def similarity(self, str1, str2):
        counter1 = Counter(list(str1))
        counter2 = Counter(list(str2))

        c = counter1 & counter2

        n = sum(c.values())
        return 2 * n / (len(str1) + len(str2))

    def learn_transformation(self, error_str, cleaned_str):
        if not cleaned_str and not error_str:
            return []

        valid_trans = [(cleaned_str, error_str)]

        l = self.longest_common_substring(cleaned_str, error_str)

        if l is None:
            return valid_trans

        lcv, rcv = cleaned_str[: l[0]], cleaned_str[l[0] + l[2] :]
        lev, rev = error_str[: l[1]], error_str[l[1] + l[2] :]

        if self.similarity(lcv, lev) + self.similarity(rcv, rev) > self.similarity(
            lcv, rev
        ) + self.similarity(rcv, lev):
            valid_trans.extend([(lcv, lev), (rcv, rev)])
            valid_trans.extend(self.learn_transformation(lcv, lev))
            valid_trans.extend(self.learn_transformation(rcv, rev))
        else:
            valid_trans.extend([(lcv, rev), (rcv, lev)])
            valid_trans.extend(self.learn_transformation(lcv, rev))
            valid_trans.extend(self.learn_transformation(rcv, lev))

        return set(valid_trans)

    def transformation_distribution(self, string_pairs):
        transforms = []
        for error_str, cleaned_str in string_pairs:
            transforms.extend(self.learn_transformation(error_str, cleaned_str))

        counter = Counter(transforms)
        sum_counter = sum(counter.values())
        return {
            transform: count * 1.0 / sum_counter for transform, count in counter.items()
        }

    def get_noisy_policy(self, str1):
        noisy_policy = {}

        for (error_str, cleaned_str), prob in self.transform_dist.items():
            if str1 in error_str:
                noisy_policy[(error_str, cleaned_str)] = prob

        sum_prob = sum(noisy_policy.values)
        noisy_policy = list(map(lambda x: (x[0], x[1] / sum_prob)))
        return noisy_policy


import json
from collections import defaultdict

from elasticsearch import Elasticsearch

from kbclean.utils.data.helpers import str2regex


class ESQuery:
    instance = None

    def __init__(self, host, port, is_async=False):
        self.es = Elasticsearch([host], port=port)

    @staticmethod
    def get_results(es_resp, field, default_value=None):
        try:
            return [x["_source"][field] for x in es_resp["hits"]["hits"]][0]
        except IndexError as e:
            return default_value

    @staticmethod
    def get_instance(host, port):
        if ESQuery.instance is None:
            ESQuery.instance = ESQuery(host, port)
        return ESQuery.instance

    def get_char_ngram_counts(self, ngrams):
        if not ngrams:
            return [0]
        query = "{}\n" + "\n{}\n".join(
            [
                json.dumps({"query": {"term": {"data": {"value": ngram}}}})
                for ngram in ngrams
            ]
        )
        return [
            ESQuery.get_results(resp, "count", default_value=0)
            for resp in self.es.msearch(query, index="char_ngram", request_timeout=60)["responses"]
        ]

    def get_tok_ngram_counts(self, ngrams):
        if not ngrams:
            return [0]
        query = "{}\n" + "\n{}\n".join(
            [
                json.dumps({"query": {"term": {"data": {"value": ngram}}}})
                for ngram in ngrams
            ]
        )
        return [
            ESQuery.get_results(resp, "count", default_value=0)
            for resp in self.es.msearch(query, index="tok_ngram", request_timeout=60)["responses"]
        ]

    def get_coexist_counts(self, values):
        set_values = set(values)
        query = "{}\n" + "\n{}\n".join(
            [
                json.dumps(
                    {
                        "query": {
                            "term": {
                                "data": {
                                    "value": str2regex(val, match_whole_token=True)
                                }
                            }
                        }
                    }
                )
                for val in set_values
            ]
        )
        mresult = self.es.msearch(query, index="n_reversed_indices")

        indices_list = [ESQuery.get_results(res, "idx") for res in mresult["responses"]]

        coexist_count = defaultdict(lambda: {})

        for idx1, val1 in enumerate(values):
            for idx2, val2 in enumerate(values):
                if indices_list[idx1] is None or indices_list[idx2] is None:
                    coexist_count[val1][val2] = 0
                else:
                    coexist_count[val1][val2] = set(indices_list[idx1]).intersection(
                        indices_list[idx2]
                    )

        return coexist_count

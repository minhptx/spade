from elasticsearch import Elasticsearch


class ESQuery:
    instance = None

    def __init__(self, host):
        self.es = Elasticsearch(host)

    @staticmethod
    def get_results(es_resp, field):
        return [x["_source"][field] for x in es_resp["hits"]["hits"]][0]

    @staticmethod
    def get_instance(host):
        if ESQuery.instance is None:
            ESQuery.instance = ESQuery(host)
        return ESQuery.instance

    def get_char_ngram_count(self, ngram):
        return ESQuery.get_results(self.es.search(
            {"query": {"term": {"data.keyword": {"value": ngram}}}}, index="char_ngram"
        ), "count")

    def get_tok_ngram_count(self, ngram):
        return ESQuery.get_results(self.es.search(
            {"query": {"term": {"data.keyword": {"value": ngram}}}}, index="tok_ngram"
        ), "count")

    # def get_coexist_count(self, str1, str2):
    #     return self.es.search(
    #         {"query": {"term": {"data.keyword": {"value": ngram}}}}, index="tok_ngram"
    #     )


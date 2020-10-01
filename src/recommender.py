from .cache import Cache, Deque
from .graphite import Graphite
from .rnn import RNN
"""The core module responsible for:
    * keeping track of person/document visits
    * learning from it
    * making recommendations
"""

class Recommender():
    """The core class

        Attributes:
            documents_n (int): Maximum number of distinct documents that are kept in memory
            persons_n (int): Maximum number of distinct persons that are kept in memory
            documents_cache (Cache): An instance of ARC cache for documents
            persons_cache (Cache): An instance of ARC cache for persons
            recs_limit (int): Maximum namber of recommendations that recommend() method returns
            rnn (RNN): Recurrent neural network model that is learnt to map a document
                to the next document visited by a person
            graphite (Graphite): graphite feeder

        The object is supposed to be created once and to be kept in memory of a recommender service
        as long as possible so that RNN can keep on improving.
        Every instantiation causes a cold-start pit, model persistence isn't supported.

        Apart from the attributes listed above, constructor takes one more optional argument:
            device (str): one of 'cuda' or 'cpu', it tells RNN model which device to use
                set it to 'cuda' if you have an Nvidia GPU available and drivers and cuDNN installed
                the end-to-end record()/recommend() curcuit when run on GTX 1050 ti works
                5 times faster than when run on Intel Core i5-4570
    """

    def __init__(
            self,
            documents_n      = 2000,
            persons_n        = 2000,
            recs_limit       = 10,
            device           = 'cpu'
        ):

        self.documents_n     = documents_n
        self.persons_n       = persons_n
        self.documents_cache = Cache(documents_n)
        self.persons_cache   = Cache(persons_n)
        self.recs_limit      = recs_limit

        # embedding dimension and hidded dimension are hardcoded
        # these numbers work well on an entertainment web-site
        # having about 1 million unique visitors per day
        # and about 5 thousand distinct pages that are being visited
        self.rnn             = RNN(documents_n, 320, 128, device)

        self.graphite        = Graphite()


    def person_history(self, person_id):
        """Looks up browsing history given a person_id
            Returns a list of zero or more document_id-s
        """
        prs_res = self.persons_cache.get_by_key(person_id)
        if prs_res is None:
            return []
        return prs_res.value.history.keys()


    def record(self, document_id, person_id):
        """Puts a visit on record
            If a person is known and they have a previous document_id in history
            and that document_id isn't equal to the current document_id
            and that document_id still exists in the documents_cache
            then RNN is learnt to map the index of the previous document
            to the index of the current document
        """

        self.graphite.send('recomlive.record_call.sum', 1)

        # Documents don't need any data in the cache but their IDs
        # therefore the second argument to get_replace() is None
        doc_res = self.documents_cache.get_replace(document_id, None)
        if doc_res.is_hit:
            # This many times another visit hit a known document
            # The ratio of document hits to visits is crucial for the quality
            # of recommendations it should remain above 90%
            # otherwise consider increasing self.documents_n
            self.graphite.send('recomlive.documents_cache_hit.sum', 1)

        # A person object has to be cached along with person_id
        # this callback creates the object when a person_id is unknown
        new_prs = lambda: Person(person_id, max(self.documents_n / 10, 10))
        # A little bit of hardcode above stands for maximum person's history length
        # which is one tenth of the documents limit but not less than ten
        prs_res = self.persons_cache.get_replace(person_id, new_prs)
        if prs_res.is_hit:
            # This hit ratio isn't super important but still nice to have an overview of it
            self.graphite.send('recomlive.persons_cache_hit.sum', 1)

        # By the way, cache.get_replace(id, [data]) will always
        # accommodate an item in the cache, therefore never returns None


        if document_id in prs_res.value.prev_recs:
            # Yay, a person "clicked" the previous recommendation!
            self.graphite.send('recomlive.recommendation_hit.sum', 1)

        prs_res.value.append_history(document_id)
        unlearned = prs_res.value.unlearned_docs()
        if len(unlearned) == 2:
            inputs = []
            for document_id in reversed(unlearned):
                doc_res = self.documents_cache.get_by_key(document_id)
                if doc_res is not None:
                    inputs.append(doc_res.idx)
            if len(inputs) >= 2:
                self.graphite.send('recomlive.rnn_learn.sum', 1)
                loss = self.rnn.fit(inputs)
                self.graphite.send('recomlive.rnn_loss.avg', loss)
                prs_res.value.mark_learned(unlearned)


    def recommend(self, document_id, person_id = None):
        """Looks up browsing history given a person_id
            Returns a list of zero or more document_id-s
        """

        self.graphite.send('recomlive.recommend_call.sum', 1)

        doc_res = self.documents_cache.get_by_key(document_id)
        if doc_res is None:
            self.graphite.send('recomlive.no_recommendations.sum', 1)
            return []

        history = {}
        prs_res = None
        if person_id is not None:
            prs_res = self.persons_cache.get_by_key(person_id)
            if prs_res is not None:
                history = prs_res.value.history

        r = self.rnn.predict(doc_res.idx)

        recs = []
        for i in r:
            if i == doc_res.idx:
                continue

            rec_res = self.documents_cache.get_by_idx(i)
            if rec_res is None:
                continue

            if rec_res.key in history:
                continue

            recs.append(rec_res.key)
            if len(recs) == self.recs_limit:
                break

        if len(recs) == 0:
            self.graphite.send('recomlive.no_recommendations.sum', 1)

        if prs_res is not None:
            prs_res.value.prev_recs = set(recs)

        return recs



class Person(object):
    def __init__(self, pid, history_max_length):
        self.id = pid
        self.history = Deque()
        self.history_max_length = history_max_length
        self.prev_recs = set()

    def append_history(self, document_id):
        if len(self.history) == self.history_max_length:
            self.history.pop()
        if len(self.history) == 0 or self.history[-1][0] != document_id:
            self.history.appendleft(document_id, False)

    def unlearned_docs(self):
        doc_ids = []
        for doc_id in self.history:
            if self.history.od[doc_id]:
                break
            doc_ids.append(doc_id)
        return doc_ids

    def mark_learned(self, doc_ids):
        for doc_id in doc_ids[1:]:
            if doc_id in self.history:
                self.history.od[doc_id] = True



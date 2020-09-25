#!/usr/bin/env python3

import sys, os
from src.server import Server
from src.recommender import Recommender

recommender = Recommender(
    int(os.getenv('RECOMMENDER_DOCS_LIMIT', 2000)),
    int(os.getenv('RECOMMENDER_PERSONS_LIMIT', 2000)),
    int(os.getenv('RECOMMENDER_RECS_LIMIT', 5)),
    os.getenv('RECOMMENDER_TORCH_DEVICE', 'cpu')
)

def dispatcher(server, data, response):
    try:
        method, did, pid = data.decode('ascii').split(',')
        if method == 'RECR':
            recommender.record(did, pid)
        elif method == 'RECM':
            recs = recommender.recommend(did, pid)
            response(pack_response('OK', recs))
        elif method == 'RR':
            recommender.record(did, pid)
            recs = recommender.recommend(did, pid)
            response(pack_response('OK', recs))
        elif method == 'PH':
            dids = recommender.person_history(pid)
            response(pack_response('OK', dids))
        else:
            raise Exception
    except:
        response(pack_response('BADMSG'))

def pack_response(status, data = []):
    return bytes(','.join([status] + data), 'ascii')


Server(
    dispatcher, port = int(os.getenv('RECOMMENDER_PORT', 25000))
).command(sys.argv[1])


#!/usr/bin/env python3

import sys, os
from src.server import Server
from src.recommender import Recommender

"""Creates recommender object, see src/recommender.py for details
"""
recommender = Recommender(
    int(os.getenv('RECOMMENDER_DOCS_LIMIT', 2000)),
    int(os.getenv('RECOMMENDER_PERSONS_LIMIT', 2000)),
    int(os.getenv('RECOMMENDER_RECS_LIMIT', 5)),
    os.getenv('RECOMMENDER_TORCH_DEVICE', 'cpu')
)

def dispatcher(server, data, response):
    """Parses request data, calls response() callback
    when there's response expected

    Args:
        server (Server): UDP server object
        data (bytes): Bytes received by the server
        response (function): A callback function,
            sends bytes back to the client
    """

    try:
        """Simple text protocol includes:
            method (str): one of RECR, RECM, RR or PH
            did (str): arbitrary document ID
            pid (str): arbitrary person ID
        """
        method, did, pid = data.decode('ascii').split(',')
        
        if method == 'RECR':
            """Records a visit: person pid visited document did
            """
            recommender.record(did, pid)

        elif method == 'RECM':
            """Makes recommendations:
                a person pid is at document did
                returns a list of dids (where to go next)
            """
            recs = recommender.recommend(did, pid)
            response(pack_response('OK', recs))

        elif method == 'RR':
            """Does both of the above
            """
            recommender.record(did, pid)
            recs = recommender.recommend(did, pid)
            response(pack_response('OK', recs))

        elif method == 'PH':
            """Returns person pid's visits history
            """
            dids = recommender.person_history(pid)
            response(pack_response('OK', dids))

        else:
            """Data is garbage
            """
            raise Exception
    except:
        response(pack_response('BADMSG'))

def pack_response(status, data = []):
    return bytes(','.join([status] + data), 'ascii')


if __name__ == '__main__':
    """Creates UDP server daemon, see src/server.py for details
    """
    server = Server(dispatcher, port = int(os.getenv('RECOMMENDER_PORT', 25000)))
    server.command(sys.argv[1])


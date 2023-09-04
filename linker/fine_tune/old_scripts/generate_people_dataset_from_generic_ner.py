import argparse
import os
from util.general import load_mongo_docs


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('-m', '--db-host', dest='db_host')
    parser.add_argument('-p', '--db-port', dest='db_port', type=int)
    parser.add_argument("-u", "--user", dest="user", default="", nargs="?")
    parser.add_argument("-r", "--replicaset", dest="replicaset", default="", nargs="?")
    return parser


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    password = os.getenv('MONGO_PASSWORD')
    ner_docs = load_mongo_docs(args.input, args.db_host, args.db_port, args.user, password, args.replicaset)
    ner_docs = [doc for doc in ner_docs if doc['answer'] == 'accept']



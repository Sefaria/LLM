from neo4j import GraphDatabase, Result, Session
from enum import Enum

class NeoExecutionType(Enum):
    WRITE = "write"
    READ = "read"


class NeoReturnType(Enum):
    SINGLE = "single"
    LIST = "list"
class QuerySettings:

    def __init__(self, execution_type: NeoExecutionType = NeoExecutionType.READ, return_type: NeoReturnType =
                 NeoReturnType.SINGLE, *return_fn_args):
        self._execution_type = execution_type
        self._return_type = return_type
        self._return_fn_args = return_fn_args

    @property
    def execution_type(self):
        return self._execution_type

    @property
    def return_type(self):
        return self._return_type

    @property
    def return_fn_args(self):
        return self._return_fn_args
class DBConnection:

    NEO_4J_URI = "bolt://localhost:7687"
    NEO_4J_USERNAME = "neo4j"
    NEO_4J_PASSWORD = "password"

    def __init__(self):
        self._driver = GraphDatabase.driver(self.NEO_4J_URI, auth=(self.NEO_4J_USERNAME, self.NEO_4J_PASSWORD))

    @property
    def driver(self):
        return self._driver

    def query(self, query, query_settings: QuerySettings, **query_kwargs):
        def transaction(tx):
            result = tx.run(query, **query_kwargs)
            return self._get_return_value(result, query_settings.return_type)

        with self.driver.session() as session:
            return self._execute_transaction(transaction, session, query_settings.execution_type)

    @staticmethod
    def _get_return_value(result: Result, return_type, *args):
        if return_type == NeoReturnType.SINGLE:
            return result.single(*args)
        if return_type == NeoReturnType.LIST:
            return result.fetch(*args)

    @staticmethod
    def _execute_transaction(transaction, session: Session, execution_type: NeoExecutionType):
        if execution_type == NeoExecutionType.WRITE:
            return session.execute_write(transaction)
        if execution_type == NeoExecutionType.READ:
            return session.execute_read(transaction)


if __name__ == '__main__':
    db = DBConnection()
    # clear DB
    db.query("MATCH (n) OPTIONAL MATCH (n)-[r]-() DELETE n,r", QuerySettings(NeoExecutionType.WRITE))

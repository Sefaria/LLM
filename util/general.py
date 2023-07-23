from db_manager import MongoProdigyDBManager


def load_mongo_docs(*db_manager_args):
    my_db = MongoProdigyDBManager(*db_manager_args)
    return my_db.output_collection.find({})

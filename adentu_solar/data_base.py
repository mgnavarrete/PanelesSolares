import pymysql


class DataBase:

    def __init__(self, host=None, port=None, user=None, password=None, database=None):
        host = 'enel-paneles-prod.colm5mlpuwae.us-east-1.rds.amazonaws.com' if host is None else host
        user = 'admin' if user is None else user
        password = 'Gary2020#' if password is None else password
        database = 'enel_paneles' if database is None else database
        port = 3306 if port is None else port
        self.connection = pymysql.connect(host=host, user=user, password=password, database=database, port=port)

    def query(self, str_qry):
        with self.connection:
            with self.connection.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(str_qry)
                result = cur.fetchall()
        return result

    def insert_query(self, str_qry):
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute(str_qry)
            self.connection.commit()


if __name__ == '__main__':
    db = DataBase()
    print(db.query('select * from panel limit 10;'))
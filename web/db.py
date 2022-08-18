import os
import json
import logging

from mysql.connector import (connection, Error)
import shortuuid

from profold2.data.parsers import parse_fasta
from web.utils import serving_log

"""
+-------------+------------------+------+-----+-------------------+----------------+
| Field       | Type             | Null | Key | Default           | Extra          |
+-------------+------------------+------+-----+-------------------+----------------+
| id          | int(10) unsigned | NO   | PRI | NULL              | auto_increment |
| job_id      | varchar(20)      | NO   | UNI | NULL              |                |
| app         | varchar(64)      | NO   |     | NULL              |                |
| email       | varchar(128)     | NO   |     | NULL              |                |
| sequences   | text             | NO   |     | NULL              |                |
| status      | varchar(10)      | NO   |     | NULL              |                |
| time_create | datetime         | YES  |     | CURRENT_TIMESTAMP |                |
| time_run    | datetime         | YES  |     | NULL              |                |
| time_done   | datetime         | YES  |     | NULL              |                |
| ip          | varchar(20)      | NO   |     | NULL              |                |
+-------------+------------------+------+-----+-------------------+----------------+

+-------------+------------------+------+-----+-------------------+----------------+
| Field       | Type             | Null | Key | Default           | Extra          |
+-------------+------------------+------+-----+-------------------+----------------+
| id          | int(10) unsigned | NO   | PRI | NULL              | auto_increment |
| description | varchar(1024)    | NO   |     | NULL              |                |
| sequence    | varchar(4096)    | NO   |     | NULL              |                |
| status      | varchar(10)      | NO   |     | NULL              |                |
| time_create | datetime         | YES  |     | CURRENT_TIMESTAMP |                |
| time_run    | datetime         | YES  |     | NULL              |                |
| time_done   | datetime         | YES  |     | NULL              |                |
| job_id      | varchar(20)      | NO   | MUL | NULL              |                |
+-------------+------------------+------+-----+-------------------+----------------+
"""

logger = logging.getLogger(__file__)

STATUS_RUNNING = 'Running'
STATUS_QUEUING = 'Queuing'
STATUS_DONE = 'Done'
STATUS_ERROR = 'Error'

APP_LIST = {'profold0': 'ProFOLD zero', 'abfold': 'AbFOLD'}
_cnx = None

def db_get():
    try:
        if _cnx:
            _cnx.ping(reconnect=True)
        else:
            raise Exception('Connect to database')
    except Exception as e:
        _cnx = connection.MySQLConnection(user='protein', password='folding',
                                 host='127.0.0.1',
                                 database='profold',
                                 autocommit=False)
    return _cnx

def app_list():
    return [dict(id=key, name=value) for key, value in APP_LIST.items()]

def app_get(app_id):
    return dict(id=app_id,
            name=APP_LIST.get(app_id, 'Unknown'))

def job_status(app_id):
    cnx = db_get()

    query = 'select status,count(*) as c from jobs where app=%s group by status'
    with cnx.cursor() as cursor:
        cursor.execute(query, (app_id,))
        for status, c in cursor:
            yield status, c

def job_get(with_tasks=True, logic_op='and', **kwargs):
    cnx = db_get()

    cond = f' {logic_op} '.join(f'{c}=%s' for c in kwargs.keys())
    query = f'select job_id,app,sequences,email,status,time_create,time_run,time_done from jobs where {cond}'
    with cnx.cursor(dictionary=True) as cursor:
        cursor.execute(query, tuple(kwargs.values()))
        job_list = cursor.fetchall()
    if job_list:
        for job in job_list:
            job_id = job['job_id']
            if with_tasks:
                query = 'select * from tasks where job_id=%s'
                with cnx.cursor(dictionary=True) as cursor:
                    cursor.execute(query, (job_id,))
                    job['tasks'] = list(map(_task_post_process,
                            cursor.fetchall()))
            log_file = serving_log(job_id)
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    job['logs'] = f.read()
    if job_list and 'job_id' in kwargs and logic_op == 'and':
        assert len(job_list) == 1
        return job_list[0]
    return job_list

def task_get(job_id, task_id, **kwargs):
    cnx = db_get()

    query = 'select * from tasks where job_id=%s and id=%s'
    with cnx.cursor(dictionary=True) as cursor:
        cursor.execute(query, (job_id, task_id))
        return _task_post_process(cursor.fetchone())

def job_set(job_id, task_id=None, **kwargs):
    cnx = db_get()

    params = ','.join([f'{c}=%s' for c in kwargs.keys()])
    cond = (job_id,)
    if task_id is None:
        query = f'update jobs set {params} where job_id=%s'
    else:
        query = f'update tasks set {params} where job_id=%s and id=%s'
        cond += (task_id,)
    logger.debug('[db.job_set] query=%s, args=%s', query, kwargs)

    with cnx.cursor(dictionary=True) as cursor:
        cursor.execute(query, tuple(kwargs.values()) + cond)
    cnx.commit()

    return job_id, task_id

def job_new(fasta_str, app, job_id=None, email=None):
    cnx = db_get()

    try:
        query = 'insert into jobs (`job_id`,`app`,`email`,`sequences`,`status`,`ip`) values (%s,%s,%s,%s,%s,%s)'
        if not job_id:
            job_id = shortuuid.uuid()[:10]
        if email is None:
            email = ''
        with cnx.cursor() as cursor:
            cursor.execute(query, (job_id, app, email, fasta_str, STATUS_QUEUING, ''))

            query = 'insert into tasks (`description`,`sequence`,`status`,`job_id`) values (%s,%s,%s,%s)'
            sequences, descriptions = parse_fasta(fasta_str)
            assert len(sequences) >= 1 and len(sequences) == len(descriptions)
            params = [(description, sequence, STATUS_QUEUING, job_id) for sequence, description in zip(sequences, descriptions)]
            cursor.executemany(query, params)

            cnx.commit()

        logger.info('job_new: commit app=%s, job_id=%s, email=%s', app, job_id, email)
    except Error as e:
        logger.error('job_new: rollback app=%s, job_id=%s, email=%s, error=%s', app, job_id, email, e)
        cnx.rollback()
        raise e

    return job_id

def _task_post_process(task):
    if task and task['metrics']:
        task['metrics'] = json.loads(task['metrics'])
    return task

if __name__ == '__main__':
    print(job_get(job_id='LVsZ3ZvHay'))
    print(job_set(job_id='LVsZ3ZvHay', status=STATUS_RUNNING))
    print(job_get(job_id='LVsZ3ZvHay'))
    print(job_set(job_id='LVsZ3ZvHay', status=STATUS_QUEUING))
    print(job_get(job_id='LVsZ3ZvHay'))

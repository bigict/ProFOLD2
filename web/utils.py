import os

def serving_dir():
    return os.path.join(
            os.environ.get('ProFOLD_HOME', '.'),
            'serving')

def serving_log(job_id):
    p = os.path.join(
            serving_dir(), 'logs')
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, f'{job_id}.log')

def serving_data(job_id):
    p = os.path.join(
            serving_dir(), 'data', job_id)
    os.makedirs(p, exist_ok=True)
    return p

def serving_pdb(job_id, task_id, prefix=''):
    return os.path.join(
            serving_data(job_id), f'{prefix}{task_id}.pdb')

def serving_svg(job_id, task_id, prefix=''):
    return os.path.join(
            serving_data(job_id), f'{prefix}{task_id}.svg')

def serving_meta(job_id, task_id, prefix=''):
    return os.path.join(
            serving_data(job_id), f'{prefix}{task_id}.log')

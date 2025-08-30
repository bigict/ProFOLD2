"""web entry point
"""
import os
import io
import zipfile

from flask import (Flask, redirect, request, render_template, send_file)

from web import db, form, utils

module = os.environ.get('ProFOLD_MODULE', '/serving')

app = Flask(__name__, static_url_path='')


@app.route('/search/', methods=['POST'])
def search():
  app_id = request.args.get('app', 'profold0')
  query = form.var_get('job_search', request.form)
  job_list = db.job_get(with_tasks=False, logic_op='or', job_id=query, email=query)
  job_status = dict(db.job_status(app_id))
  return render_template(
      'search.html',
      module=module,
      app=db.app_get(app_id),
      app_list=db.app_list(),
      job_search=query,
      job_list=job_list,
      job_status=job_status
  )


@app.route('/checking/<job_id>/', methods=['GET'])
def checking(job_id):
  refresh = request.args.get('refresh', 30)
  app_id = request.args.get('app', 'profold0')

  job = db.job_get(job_id=job_id)
  if not job:
    return redirect(f'{module}/error?app={app_id}&id={job_id}')
  #if job['status'] == 'Done':
  #    return redirect(f'result?app={app}&id={job_id}')

  job_status = dict(db.job_status(app_id))
  return render_template(
      'checking.html',
      module=module,
      refresh=refresh,
      app=db.app_get(app_id),
      app_list=db.app_list(),
      job=job,
      job_status=job_status
  )


@app.route('/result/<job_id>/<task_id>/', methods=['GET'])
def result(job_id, task_id):
  app_id = request.args.get('app', 'profold0')
  job_status = dict(db.job_status(app_id))
  return render_template(
      'result.html',
      module=module,
      app=db.app_get(app_id),
      app_list=db.app_list(),
      job=db.job_get(job_id=job_id),
      task=db.task_get(job_id=job_id, task_id=task_id),
      job_status=job_status
  )


@app.route('/resource/<job_id>/<data>/', methods=['GET'], defaults={'task_id': None})
@app.route('/resource/<job_id>/<task_id>/<data>/', methods=['GET'])
def resource(data, job_id, task_id=None):
  def task_zip(f, task):
    task_id, description, sequence = (task['id'], task['description'], task['sequence'])
    f.writestr(f'{task_id}.fasta', f'>{description}\n{sequence}')

    # pdb
    p = utils.serving_pdb(job_id, task_id)
    if not os.path.exists(p):
      return redirect(f'{module}/error?app={app_id}')
    f.write(p, arcname=os.path.basename(p))
    # relax_pdb
    p = utils.serving_pdb(job_id, task_id, prefix='relax_')
    if not os.path.exists(p):
      return redirect(f'{module}/error?app={app_id}')
    f.write(p, arcname=os.path.basename(p))
    # svg
    p = utils.serving_svg(job_id, task_id)
    if not os.path.exists(p):
      return redirect(f'{module}/error?app={app_id}')
    f.write(p, arcname=os.path.basename(p))

  app_id = request.args.get('app', 'profold0')
  mimetype = None
  if task_id is None:
    if data == 'zip':
      job = db.job_get(job_id=job_id, app=app_id)
      if not job or 'tasks' not in job or not job['tasks']:
        return redirect(f'{module}/error?app={app_id}')
      with io.BytesIO() as f:
        with zipfile.ZipFile(f, 'w') as obj:
          for task in job['tasks']:
            task_zip(obj, task)
        return send_file(
            io.BytesIO(f.getvalue()),
            mimetype='application/zip, application/octet-stream',
            as_attachment=True,
            attachment_filename=f'{job_id}.zip'
        )
  else:
    if data == 'pdb':
      p = utils.serving_pdb(job_id, task_id)
    elif data == 'relaxed_pdb':
      p = utils.serving_pdb(job_id, task_id, prefix='relax_')
    elif data == 'svg':
      p, mimetype = utils.serving_svg(job_id, task_id), 'image/svg+xml'
    elif data == 'zip':
      task = db.task_get(job_id=job_id, task_id=task_id)
      if not task:
        return redirect(f'{module}/error?app={app_id}')
      with io.BytesIO() as f:
        with zipfile.ZipFile(f, 'w') as obj:
          task_zip(obj, task)
        return send_file(
            io.BytesIO(f.getvalue()),
            mimetype='application/zip, application/octet-stream',
            as_attachment=True,
            attachment_filename=f'{job_id}_{task_id}.zip'
        )
    else:
      return redirect(f'{module}/error?app={app_id}')

    if os.path.exists(p):
      return send_file(p, mimetype=mimetype, as_attachment=True)

  return redirect(f'{module}/error?app={app_id}')


@app.route('/diff/<exp_name>/', methods=['GET'])
def diff(exp_name):
  diff_list = []
  num_pred = 0

  static_folder = app.static_folder if app.has_static_folder else 'static'
  exp_file = os.path.join(static_folder, f'{exp_name}.diff')
  if os.path.exists(exp_file):
    with open(exp_file, 'r') as f:
      for line in filter(lambda x: len(x) > 0, map(lambda x: x.strip(), f)):
        desc, seq, truth_pdb, truth_svg, pred_list = line.split('\t', 4)
        if desc.startswith('>'):
          desc = desc[1:]
        pid = desc.split()[0]
        pred_list = pred_list.split('\t')
        assert len(pred_list) % 3 == 0
        pred_list = [
            dict(
                desc=pred_list[3 * i],
                pred_pdb=pred_list[2 * i + 1],
                pred_svg=pred_list[2 * i + 2]
            ) for i in range(len(pred_list) // 3)
        ]
        if num_pred == 0:
          num_pred = len(pred_list)
        assert len(pred_list) == num_pred
        diff_list.append(
            dict(
                pid=pid,
                desc=desc,
                seq=seq,
                pred_list=pred_list,
                truth_pdb=truth_pdb,
                truth_svg=truth_svg
            )
        )

  return render_template(
      'diff.html',
      module=module,
      exp_name=exp_name,
      num_pred=num_pred,
      diff_list=diff_list
  )


@app.route('/error/', methods=['GET'])
def error():
  app_id = request.args.get('app', 'profold0')
  job_status = dict(db.job_status(app_id))
  return render_template(
      'error.html', module=module, app=db.app_get(app_id), job_status=job_status
  )


@app.route('/', defaults={'job_id': None}, methods=['GET', 'POST'])
@app.route('/index/', defaults={'job_id': None}, methods=['GET', 'POST'])
@app.route('/submit/<job_id>/', methods=['GET', 'POST'])
def submit(job_id):
  app_id = request.args.get('app', 'profold0')
  job_status = dict(db.job_status(app_id))
  if request.method == 'GET':
    if job_id:
      job = db.job_get(with_tasks=False, job_id=job_id, app=app_id)
      del job['job_id']
    else:
      job = None
    return render_template(
        'index.html',
        module=module,
        args=job,
        app=db.app_get(app_id),
        app_list=db.app_list(),
        job_status=job_status
    )
  elif request.method == 'POST':
    errors = form.validate(request.values, request.files)
    if errors:
      return render_template(
          'index.html',
          module=module,
          args=request.values,
          app=db.app_get(app_id),
          app_list=db.app_list(),
          job_status=job_status,
          errors=errors
      )
    job_id = db.job_new(
        form.var_get(
            ('sequences', 'sequence_file'), request.values, files=request.files
        ),
        app_id,
        job_id=form.var_get('job_id', request.values),
        email=form.var_get('email', request.values)
    )
    return redirect(f'{module}/checking/{job_id}/?app={app_id}')


if __name__ == '__main__':
  app.config['DEBUG'] = True
  app.run(host='0.0.0.0', port=9090, threaded=True)

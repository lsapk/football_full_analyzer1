from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os, subprocess, uuid
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(),'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
RESULTS_FOLDER = os.path.join(os.getcwd(),'web_results')
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='POST':
        f = request.files.get('video')
        if not f:
            return 'No file', 400
        uid = str(uuid.uuid4())[:8]
        save_path = os.path.join(UPLOAD_FOLDER, uid + '_' + f.filename)
        f.save(save_path)
        outdir = os.path.join(RESULTS_FOLDER, uid)
        os.makedirs(outdir, exist_ok=True)
        # launch processing as background process (simple)
        cmd = ['python', os.path.join(os.getcwd(),'main.py'), '--video', save_path, '--output', outdir]
        subprocess.Popen(cmd)
        return redirect(url_for('status', jobid=uid))
    return render_template('index.html')

@app.route('/status/<jobid>')
def status(jobid):
    outdir = os.path.join(RESULTS_FOLDER, jobid)
    exists = os.path.exists(outdir)
    files = os.listdir(outdir) if exists else []
    return render_template('status.html', jobid=jobid, files=files)

@app.route('/results/<jobid>/<path:filename>')
def results(jobid, filename):
    outdir = os.path.join(RESULTS_FOLDER, jobid)
    return send_from_directory(outdir, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

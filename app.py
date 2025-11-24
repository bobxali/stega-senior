
from flask import Flask, render_template, request, flash, send_file, redirect, url_for
from PIL import Image
import algorithms.lsb as lsb_mod
import algorithms.pvd as pvd_mod
import algorithms.dct as dct_mod
import os
import uuid

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'dev_secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        image_file = request.files.get('image')
        message = request.form.get('message', '')
        algorithm = request.form.get('algorithm', 'LSB')

        if not image_file or image_file.filename == '':
            flash('No image uploaded')
            return redirect(request.url)

        try:
            img = Image.open(image_file.stream)
        except Exception:
            flash('Uploaded file is not a valid image')
            return redirect(request.url)

        # choose algorithm module
        if algorithm == 'LSB':
            alg = lsb_mod
        elif algorithm == 'PVD':
            alg = pvd_mod
        elif algorithm == 'DCT':
            alg = dct_mod
        else:
            flash('Unknown algorithm selected')
            return redirect(request.url)

        try:
            payload = message.encode('utf-8')
            encoded_img = alg.encode_pil(img, payload)
        except Exception as e:
            flash(f'Encoding failed: {e}')
            return redirect(request.url)

        # Save encoded image with a unique name
        out_name = f"encoded_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        try:
            encoded_img.save(out_path)
        except Exception as e:
            flash(f'Failed to save encoded image: {e}')
            return redirect(request.url)

        # Return the file as a download
        return send_file(out_path, as_attachment=True)

    return render_template('index.html')


@app.route('/decode', methods=['GET','POST'])
def decode():
    if request.method == 'POST':
        image_file = request.files.get('image')
        algorithm = request.form.get('algorithm', 'AUTO')

        if not image_file or image_file.filename == '':
            flash('No image uploaded')
            return redirect(request.url)

        try:
            img = Image.open(image_file.stream)
        except Exception:
            flash('Uploaded file is not a valid image')
            return redirect(request.url)

        # Auto-detect algorithm if selected
        if algorithm == 'AUTO':
            results = []
            for alg_name, alg_mod in [('LSB', lsb_mod), ('PVD', pvd_mod), ('DCT', dct_mod)]:
                try:
                    data = alg_mod.decode_pil(img)
                    try:
                        text = data.decode('utf-8')
                        results.append((alg_name, text, True))
                    except Exception:
                        results.append((alg_name, repr(data), False))
                except Exception as e:
                    pass
            
            if results:
                # Prefer valid UTF-8 results; otherwise use first available
                valid_results = [r for r in results if r[2]]
                if valid_results:
                    alg_name, text, _ = valid_results[0]
                else:
                    alg_name, text, _ = results[0]
                return render_template('decode.html', message=text, algorithm=alg_name)
            else:
                flash('Could not decode with any algorithm')
                return redirect(request.url)
        else:
            # Specific algorithm selected
            if algorithm == 'LSB':
                alg = lsb_mod
            elif algorithm == 'PVD':
                alg = pvd_mod
            elif algorithm == 'DCT':
                alg = dct_mod
            else:
                flash('Unknown algorithm selected')
                return redirect(request.url)

            try:
                data = alg.decode_pil(img)
            except Exception as e:
                flash(f'Decoding failed: {e}')
                return redirect(request.url)

            try:
                text = data.decode('utf-8')
            except Exception:
                text = repr(data)

            return render_template('decode.html', message=text, algorithm=algorithm)

    return render_template('decode.html', message=None, algorithm=None)

if __name__ == '__main__':
    app.run(debug=True)

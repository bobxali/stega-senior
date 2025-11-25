
from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
from PIL import Image
import algorithms.lsb as lsb_mod
import algorithms.pvd as pvd_mod
import algorithms.dct as dct_mod
from algorithms.capacity import StegoCapacity
import os
import uuid
import io
import numpy as np
import math
import cv2

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'dev_secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def _compute_psnr(orig: Image.Image, stego: Image.Image) -> float:
    a = np.asarray(orig.convert("RGB"), dtype=np.float64)
    b = np.asarray(stego.convert("RGB"), dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def _compute_ssim(orig: Image.Image, stego: Image.Image) -> float:
    # Grayscale SSIM using Gaussian window (closer to reference implementation)
    x = np.asarray(orig.convert("L"), dtype=np.float64)
    y = np.asarray(stego.convert("L"), dtype=np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu_x = cv2.filter2D(x, -1, window)
    mu_y = cv2.filter2D(y, -1, window)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = cv2.filter2D(x * x, -1, window) - mu_x2
    sigma_y2 = cv2.filter2D(y * y, -1, window) - mu_y2
    sigma_xy = cv2.filter2D(x * y, -1, window) - mu_xy

    L = 255
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean())

def _save_upload(file_storage):
    ext = os.path.splitext(file_storage.filename)[1] or '.png'
    fname = f"upload_{uuid.uuid4().hex}{ext}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file_storage.save(path)
    img = Image.open(path)
    return fname, img


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        image_file = request.files.get('image')
        message = request.form.get('message', '')
        algorithm = request.form.get('algorithm', 'LSB')
        action = request.form.get('action', 'encode')
        existing_image = request.form.get('current_image')
        stored_name = None
        img = None

        # Determine which image to use: new upload or existing stored file
        if image_file and image_file.filename:
            try:
                stored_name, img = _save_upload(image_file)
            except Exception:
                flash('Uploaded file is not a valid image')
                return redirect(request.url)
        elif existing_image:
            stored_name = existing_image
            stored_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_name)
            if not os.path.exists(stored_path):
                flash('Previously uploaded image not found')
                return redirect(request.url)
            try:
                img = Image.open(stored_path)
            except Exception:
                flash('Stored image is not readable')
                return redirect(request.url)
        else:
            flash('No image uploaded')
            return redirect(request.url)

        if action == 'clear':
            if stored_name:
                try:
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], stored_name))
                except Exception:
                    pass
            return redirect(url_for('index'))

        if action == 'capacity':
            caps = StegoCapacity.for_all(img, len(lsb_mod.DELIMITER), dct_redundancy=dct_mod.REDUNDANCY)
            return render_template('index.html', capacity_info=caps, current_image=stored_name, message_value=message, selected_algorithm=algorithm)

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
        except ValueError as e:
            caps = StegoCapacity.for_all(img, len(lsb_mod.DELIMITER), dct_redundancy=dct_mod.REDUNDANCY)
            flash(f'Encoding failed: {e}')
            return render_template('index.html', capacity_info=caps, current_image=stored_name, message_value=message, selected_algorithm=algorithm)
        except Exception as e:
            flash(f'Encoding failed: {e}')
            return redirect(request.url)

        # Save encoded image with a unique name
        alg_label = algorithm.lower()
        out_name = f"{alg_label}_encoded_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        try:
            encoded_img.save(out_path)
        except Exception as e:
            flash(f'Failed to save encoded image: {e}')
            return redirect(request.url)

        # Compute quality metrics
        psnr = _compute_psnr(img, encoded_img)
        ssim = _compute_ssim(img, encoded_img)

        return render_template(
            'index.html',
            encoded_image=out_name,
            current_image=stored_name,
            message_value=message,
            selected_algorithm=algorithm,
            metrics_info={'psnr': psnr, 'ssim': ssim}
        )

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

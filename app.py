
from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
from PIL import Image
import algorithms.lsb as lsb_mod
import algorithms.pvd as pvd_mod
import algorithms.dct as dct_mod
import algorithms.steganalysis as steganalysis_mod
from algorithms.capacity import StegoCapacity
from werkzeug.utils import secure_filename
import os
import uuid
import io
import numpy as np
import math
import cv2

UPLOAD_FOLDER = 'uploads'
IMAGES_FOLDER = 'images'  # optional library of cover images
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)
FILE_MAGIC = b"FILEV1"  # marker for file payloads

app = Flask(__name__)
app.secret_key = 'dev_secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Peak signal-to-noise ratio between two RGB images
def _compute_psnr(orig: Image.Image, stego: Image.Image) -> float:
    a = np.asarray(orig.convert("RGB"), dtype=np.float64)
    b = np.asarray(stego.convert("RGB"), dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# Grayscale SSIM using Gaussian window (closer to reference implementation)
def _compute_ssim(orig: Image.Image, stego: Image.Image) -> float:
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


# Save upload to disk and return (stored name, opened image)
def _save_upload(file_storage):
    ext = os.path.splitext(file_storage.filename)[1] or '.png'
    fname = f"upload_{uuid.uuid4().hex}{ext}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file_storage.save(path)
    img = Image.open(path)
    return fname, img


def _build_payload(payload_type: str, message: str, file_storage):
    """
    Build the outbound payload bytes depending on mode.
    Returns (payload_bytes, file_label) where file_label is None for text.
    """
    if payload_type == 'file':
        if not file_storage or not file_storage.filename:
            raise ValueError("No file provided for file payload mode")
        fname = secure_filename(file_storage.filename) or "attachment.bin"
        name_bytes = fname.encode("utf-8", errors="ignore")
        if len(name_bytes) > 65535:
            raise ValueError("Filename too long")
        file_bytes = file_storage.read()
        # Reset stream position so subsequent reads (if any) still work
        try:
            file_storage.stream.seek(0)
        except Exception:
            pass
        header = FILE_MAGIC + len(name_bytes).to_bytes(2, "big") + name_bytes
        return header + file_bytes, fname
    # default text
    return message.encode("utf-8"), None


def _parse_payload(data: bytes):
    """
    Interpret embedded payload.
    Returns dict with type: 'file' or 'text', plus fields.
    """
    if data.startswith(FILE_MAGIC) and len(data) >= len(FILE_MAGIC) + 2:
        name_len = int.from_bytes(data[len(FILE_MAGIC):len(FILE_MAGIC)+2], "big")
        start = len(FILE_MAGIC) + 2
        if 0 <= name_len <= 65535 and len(data) >= start + name_len:
            name = data[start:start + name_len].decode("utf-8", errors="ignore") or "extracted.bin"
            content = data[start + name_len:]
            return {"type": "file", "filename": name, "bytes": content}
    try:
        text = data.decode("utf-8")
        return {"type": "text", "text": text, "utf8": True}
    except Exception:
        return {"type": "text", "text": repr(data), "utf8": False}


def _list_library_images():
    images = []
    for fname in os.listdir(IMAGES_FOLDER):
        if not fname.lower().endswith((".png", ".bmp", ".jpg", ".jpeg", ".tiff", ".tif")):
            continue
        path = os.path.join(IMAGES_FOLDER, fname)
        try:
            img = Image.open(path)
            w, h = img.size
            cap = StegoCapacity.lsb_bytes(img, len(lsb_mod.DELIMITER))
            images.append({"filename": fname, "width": w, "height": h, "capacity": cap})
        except Exception:
            continue
    # Sort by capacity descending then name
    images.sort(key=lambda x: (-x["capacity"], x["filename"]))
    return images


def _import_library_image(fname: str):
    if not fname:
        raise ValueError("No library image selected")
    safe_name = secure_filename(fname)
    src_path = os.path.join(IMAGES_FOLDER, safe_name)
    if not os.path.exists(src_path):
        raise ValueError("Selected library image not found")
    img = Image.open(src_path)
    ext = os.path.splitext(safe_name)[1] or '.png'
    new_name = f"lib_{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(app.config['UPLOAD_FOLDER'], new_name)
    img.save(dest_path)
    return new_name, img


def _recommend_algorithm(img: Image.Image, payload_len: int):
    """
    Simple heuristic recommender:
    - Filters algorithms by capacity
    - Scores predicted quality using payload fill ratio (lower is better)
    - Returns best algorithm and a human-readable reason
    """
    caps = {
        "LSB": StegoCapacity.lsb_bytes(img, len(lsb_mod.DELIMITER)),
        "PVD": StegoCapacity.pvd_bytes(img),
        "DCT": StegoCapacity.dct_bytes(img, redundancy=dct_mod.REDUNDANCY),
    }
    candidates = []
    for name, cap in caps.items():
        if cap <= 0 or payload_len > cap:
            continue
        ratio = payload_len / cap
        # Predict quality: start high and drop with ratio; DCT is more stable
        if name == "LSB":
            psnr_est = 55 - 25 * ratio
        elif name == "PVD":
            psnr_est = 52 - 20 * ratio
        else:  # DCT
            psnr_est = 50 - 15 * ratio
        candidates.append((psnr_est, ratio, name, cap))

    if not candidates:
        return None

    # Sort by estimated quality then lower ratio (more headroom)
    candidates.sort(key=lambda x: (-x[0], x[1]))
    best = candidates[0]
    psnr_est, ratio, name, cap = best
    reason = f"{name} fits ({payload_len} / {cap} bytes); predicted quality ~{psnr_est:.1f} dB"
    return {
        "algorithm": name,
        "reason": reason,
        "estimated_psnr": psnr_est,
        "fill_ratio": ratio,
        "capacities": caps,
    }


def _recommend_image_for_payload(base_dir: str, payload_len: int):
    """
    Scan uploads for candidate images and return the filename with the largest LSB capacity that fits.
    """
    best = None
    for fname in os.listdir(base_dir):
        if fname.lower().endswith((".png", ".bmp", ".jpg", ".jpeg", ".tiff", ".tif")):
            path = os.path.join(base_dir, fname)
            try:
                img = Image.open(path)
                cap = StegoCapacity.lsb_bytes(img, len(lsb_mod.DELIMITER))
                if payload_len <= cap:
                    if best is None or cap > best[1]:
                        best = (fname, cap)
            except Exception:
                continue
    if best:
        return {"filename": best[0], "capacity": best[1]}
    return None


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/images/<path:filename>')
def library_file(filename):
    return send_from_directory(IMAGES_FOLDER, filename)


# Main page: upload/retain image, capacity check, encode payload, show metrics
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        available_images = _list_library_images()
        image_file = request.files.get('image')
        message = request.form.get('message', '')
        algorithm = request.form.get('algorithm', 'LSB')
        action = request.form.get('action', 'encode')
        key = (request.form.get('key') or '').strip() or None
        payload_type = request.form.get('payload_type', 'text')
        payload_file = request.files.get('payload_file')
        existing_image = request.form.get('current_image')
        library_image = request.form.get('library_image')
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
        elif library_image:
            try:
                stored_name, img = _import_library_image(library_image)
            except Exception as e:
                flash(str(e))
                return redirect(request.url)
        else:
            flash('No image uploaded')
            return redirect(request.url)

        # Build payload early to reuse for recommend/encode paths
        try:
            payload, file_label = _build_payload(payload_type, message, payload_file)
        except ValueError as e:
            flash(str(e))
            return redirect(request.url)
        except Exception as e:
            flash(f'Failed to prepare payload: {e}')
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
            return render_template('index.html', capacity_info=caps, current_image=stored_name, message_value=message, selected_algorithm=algorithm, key_value=key, payload_type=payload_type, available_images=available_images, library_selected=library_image)

        if action == 'recommend':
            rec = _recommend_algorithm(img, len(payload))
            suggested_image = None
            suggestion_note = None
            if not rec:
                suggested_image = _recommend_image_for_payload(app.config['UPLOAD_FOLDER'], len(payload))
            else:
                # If current image is tight, suggest a bigger one
                if rec["fill_ratio"] > 0.7:
                    suggested_image = _recommend_image_for_payload(app.config['UPLOAD_FOLDER'], len(payload))
                if rec["fill_ratio"] > 0.9:
                    suggestion_note = "Payload nearly fills capacity; choose a larger image for better quality."
            caps = rec["capacities"] if rec else None
            return render_template(
                'index.html',
                capacity_info=caps,
                current_image=stored_name,
                message_value=message,
                selected_algorithm=rec["algorithm"] if rec else algorithm,
                key_value=key,
                payload_type=payload_type,
                recommendation=rec,
                suggested_image=suggested_image,
                suggestion_note=suggestion_note,
                available_images=available_images,
                library_selected=library_image,
                file_label=file_label if payload_type == 'file' else None,
            )

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
            encoded_img = alg.encode_pil(img, payload, key=key)
        except ValueError as e:
            caps = StegoCapacity.for_all(img, len(lsb_mod.DELIMITER), dct_redundancy=dct_mod.REDUNDANCY)
            flash(f'Encoding failed: {e}')
            return render_template('index.html', capacity_info=caps, current_image=stored_name, message_value=message, selected_algorithm=algorithm, key_value=key, payload_type=payload_type, available_images=available_images, library_selected=library_image)
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
            key_value=key,
            payload_type=payload_type,
            file_label=file_label if payload_type == 'file' else None,
            available_images=available_images,
            library_selected=library_image,
            metrics_info={'psnr': psnr, 'ssim': ssim}
        )

    return render_template('index.html', key_value=None, payload_type='text', available_images=_list_library_images(), library_selected=None)


# Decode page: optionally auto-detect algorithm, otherwise honor selection
@app.route('/decode', methods=['GET','POST'])
def decode():
    if request.method == 'POST':
        image_file = request.files.get('image')
        algorithm = request.form.get('algorithm', 'AUTO')
        key = (request.form.get('key') or '').strip() or None

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
                    data = alg_mod.decode_pil(img, key=key)
                    parsed = _parse_payload(data)
                    results.append((alg_name, parsed))
                except Exception as e:
                    pass
            
            if results:
                # Prefer files, then valid UTF-8 text, otherwise first available
                file_results = [r for r in results if r[1].get("type") == "file"]
                utf8_results = [r for r in results if r[1].get("type") == "text" and r[1].get("utf8")]
                if file_results:
                    alg_name, parsed = file_results[0]
                elif utf8_results:
                    alg_name, parsed = utf8_results[0]
                else:
                    alg_name, parsed = results[0]

                if parsed.get("type") == "file":
                    safe_name = os.path.basename(parsed.get("filename") or "extracted.bin")
                    out_name = f"extracted_{uuid.uuid4().hex}_{safe_name}"
                    out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
                    with open(out_path, "wb") as f:
                        f.write(parsed.get("bytes", b""))
                    return render_template('decode.html', message=None, algorithm=alg_name, key_value=key, file_download=out_name, file_label=safe_name)

                return render_template('decode.html', message=parsed.get("text"), algorithm=alg_name, key_value=key)
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
                data = alg.decode_pil(img, key=key)
            except Exception as e:
                flash(f'Decoding failed: {e}')
                return redirect(request.url)

            parsed = _parse_payload(data)
            if parsed.get("type") == "file":
                safe_name = os.path.basename(parsed.get("filename") or "extracted.bin")
                out_name = f"extracted_{uuid.uuid4().hex}_{safe_name}"
                out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
                with open(out_path, "wb") as f:
                    f.write(parsed.get("bytes", b""))
                return render_template('decode.html', message=None, algorithm=algorithm, key_value=key, file_download=out_name, file_label=safe_name)

            return render_template('decode.html', message=parsed.get("text"), algorithm=algorithm, key_value=key)

    return render_template('decode.html', message=None, algorithm=None, key_value=None)


@app.route('/steganalysis', methods=['GET','POST'])
def steganalysis():
    analysis = None
    image_name = None
    if request.method == 'POST':
        image_file = request.files.get('image')
        if not image_file or image_file.filename == '':
            flash('No image uploaded')
            return redirect(request.url)
        try:
            img = Image.open(image_file.stream)
            analysis = steganalysis_mod.analyze_pil(img)
            image_name = image_file.filename
        except Exception as e:
            flash(f'Could not analyze image: {e}')
            return redirect(request.url)
    return render_template('steganalysis.html', analysis=analysis, image_name=image_name)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template, redirect, url_for, jsonify, session
import cv2
import numpy as np
import os
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ANSWER_KEY_FOLDER = 'static/answer_keys'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['ANSWER_KEY_FOLDER'] = ANSWER_KEY_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(ANSWER_KEY_FOLDER, exist_ok=True)

DEFAULT_PARAMS = {
    'threshold': 127,
    'min_area': 150,
    'max_area': 1000,
    'questions': 60,
    'choices': 4,
    'zoom': 100,
    'width': None,
    'height': None,
    'x_offset': 0,
    'y_offset': 0,
    'parts': 1,
    'answer_key': None,
    'answer_key_name': None
}


def process_omr(image_path, params):
    image = cv2.imread(image_path)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, params['threshold'], 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if params['min_area'] < cv2.contourArea(cnt) < params['max_area']]

    valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[1])

    correct_answers = 0
    student_answers = []

    # Column based no of parts
    part_width = image.shape[1] // params['parts']

    for part in range(params['parts']):
        part_contours = [cnt for cnt in valid_contours if part * part_width <= cv2.boundingRect(cnt)[0] <
                         (part + 1) * part_width]
        part_contours = sorted(part_contours, key=lambda c: cv2.boundingRect(c)[1])

        # Process each row (question)
        for i in range(0, min(len(part_contours), params['questions'] * params['choices']), params['choices']):
            question_contours = part_contours[i:i + params['choices']]

            # Sort contours(left to right)
            question_contours = sorted(question_contours, key=lambda c: cv2.boundingRect(c)[0])

            if len(question_contours) != params['choices']:
                continue

            bubble_areas = []
            for j, bubble in enumerate(question_contours):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [bubble], -1, 255, -1)

                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                area = cv2.countNonZero(mask)
                bubble_areas.append((area, j))

            bubble_areas = sorted(bubble_areas, reverse=True)

            # multiple answers or blank answers
            max_area, max_idx = bubble_areas[0]
            second_max_area = bubble_areas[1][0] if len(bubble_areas) > 1 else 0

            if max_area == 0:
                max_idx = -1
            elif max_area > 0 and second_max_area > 0.5 * max_area:
                max_idx = -1

            student_answers.append(max_idx)

            # Check if answer is correct
            question_number = (i // params['choices']) + (part * params['questions'] // params['parts'])
            if question_number < len(params['answer_key']):
                is_correct = (max_idx == params['answer_key'][question_number])
                color = (0, 255, 0) if is_correct else (0, 0, 255)

                correct_answers += 1 if is_correct else 0

                if max_idx != -1:
                    cv2.drawContours(image, [question_contours[max_idx]], -1, color, 3)
                    cv2.putText(image, str(max_idx),
                                (question_contours[max_idx][0][0][0] - 10, question_contours[max_idx][0][0][1] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    score_text = f"Score: {correct_answers}/{params['questions']}"
    cv2.putText(image, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    result_path = os.path.join(app.config['RESULT_FOLDER'], 'omr_result.jpg')
    cv2.imwrite(result_path, image)
    return result_path, correct_answers, student_answers, params, score_text

    # cv2.imwrite("omr_result.jpg", image)
    # image = cv2.imread(image_path)
    # original = image.copy()
    #
    # if params['width'] is None:
    #     params['width'] = image.shape[1]
    # if params['height'] is None:
    #     params['height'] = image.shape[0]
    #
    # zoom_factor = params['zoom'] / 100.0
    # x, y = params['x_offset'], params['y_offset']
    # w, h = int(params['width'] * zoom_factor), int(params['height'] * zoom_factor)
    #
    # zoomed_image = cv2.resize(original, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
    # x_end = min(x + w, zoomed_image.shape[1])
    # y_end = min(y + h, zoomed_image.shape[0])
    # cropped = zoomed_image[y:y_end, x:x_end]
    #
    # gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, params['threshold'], 255, cv2.THRESH_BINARY_INV)
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # valid_contours = [cnt for cnt in contours if params['min_area'] < cv2.contourArea(cnt) < params['max_area']]
    # valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[1])
    #
    # answer_key = params['answer_key'] or [i % params['choices'] for i in range(params['questions'])]
    # correct_answers = 0
    # student_answers = []
    # answer_details = []
    # image_display = cropped.copy()
    #
    # part_width = cropped.shape[1] // params['parts']
    # total_questions = min(params['questions'], len(answer_key))
    #
    # for part in range(params['parts']):
    #     part_contours = [cnt for cnt in valid_contours if
    #                      part * part_width <= cv2.boundingRect(cnt)[0] < (part + 1) * part_width]
    #     part_contours = sorted(part_contours, key=lambda c: cv2.boundingRect(c)[1])
    #
    #     for i in range(0, min(len(part_contours), total_questions * params['choices']), params['choices']):
    #         question_contours = part_contours[i:i + params['choices']]
    #         question_contours = sorted(question_contours, key=lambda c: cv2.boundingRect(c)[0])
    #
    #         if len(question_contours) != params['choices']:
    #             continue
    #
    #         bubble_areas = []
    #         for j, bubble in enumerate(question_contours):
    #             mask = np.zeros(thresh.shape, dtype="uint8")
    #             cv2.drawContours(mask, [bubble], -1, 255, -1)
    #             mask = cv2.bitwise_and(thresh, thresh, mask=mask)
    #             area = cv2.countNonZero(mask)
    #             bubble_areas.append((area, j))
    #
    #         bubble_areas = sorted(bubble_areas, reverse=True)
    #         max_area, max_idx = bubble_areas[0]
    #         second_max_area = bubble_areas[1][0] if len(bubble_areas) > 1 else 0
    #
    #         if max_area == 0:
    #             max_idx = -1
    #         elif max_area > 0 and second_max_area > 0.5 * max_area:
    #             max_idx = -1
    #
    #         student_answers.append(max_idx)
    #         question_number = (i // params['choices']) + (part * total_questions // params['parts'])
    #
    #         if question_number < len(answer_key):
    #             is_correct = (max_idx == answer_key[question_number])
    #             color = (0, 255, 0) if is_correct else (0, 0, 255)
    #             correct_answers += 1 if is_correct else 0
    #
    #             if max_idx != -1:
    #                 cv2.drawContours(image_display, [question_contours[max_idx]], -1, color, 3)
    #                 cv2.putText(image_display, str(max_idx),
    #                             (question_contours[max_idx][0][0][0] - 10,
    #                              question_contours[max_idx][0][0][1] + 10),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #
    #             # In the answer_details.append() call
    #             answer_details.append({
    #                 'question': question_number + 1,
    #                 'selected': max_idx,
    #                 'correct': answer_key[question_number],
    #                 'is_correct': is_correct
    #                 # Removed: 'confidence': max_area / (params['max_area'] * 0.5) if params['max_area'] > 0 else 0
    #             })
    #
    # accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    # score_text = f"Score: {correct_answers}/{total_questions} ({accuracy:.1f}%)"
    # cv2.putText(image_display, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    #
    # result_path = os.path.join(app.config['RESULT_FOLDER'], 'omr_result.jpg')
    # cv2.imwrite(result_path, image_display)

    # pass
    # return result_path, correct_answers, accuracy, answer_details, params, student_answers


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        params = {
            'threshold': int(request.form.get('threshold', DEFAULT_PARAMS['threshold'])),
            'min_area': int(request.form.get('min_area', DEFAULT_PARAMS['min_area'])),
            'max_area': int(request.form.get('max_area', DEFAULT_PARAMS['max_area'])),
            'questions': int(request.form.get('questions', DEFAULT_PARAMS['questions'])),
            'choices': int(request.form.get('choices', DEFAULT_PARAMS['choices'])),
            'zoom': int(request.form.get('zoom', DEFAULT_PARAMS['zoom'])),
            'width': int(request.form.get('width', DEFAULT_PARAMS['width'])) if 'width' in request.form else None,
            'height': int(request.form.get('height', DEFAULT_PARAMS['height'])) if 'height' in request.form else None,
            'x_offset': int(request.form.get('x_offset', DEFAULT_PARAMS['x_offset'])),
            'y_offset': int(request.form.get('y_offset', DEFAULT_PARAMS['y_offset'])),
            'parts': int(request.form.get('parts', DEFAULT_PARAMS['parts'])),
            'answer_key': session.get('current_answer_key'),
            'answer_key_name': session.get('current_answer_key_name')
        }

        result_path, correct_answers, student_answers, params, score_text= process_omr(filepath, params)
        return render_template('result.html',
                               result_image=result_path,
                               score=score_text,
                               total_questions=params['questions'],
                               # accuracy=accuracy,
                               # answer_details=answer_details,
                               answer_key_name=params['answer_key_name'],
                               student_answers=student_answers)

    answer_keys = []
    if os.path.exists(app.config['ANSWER_KEY_FOLDER']):
        answer_keys = [f for f in os.listdir(app.config['ANSWER_KEY_FOLDER']) if f.endswith('.json')]

    return render_template('index2.html', params=DEFAULT_PARAMS, answer_keys=answer_keys)


@app.route('/save_answer_key', methods=['POST'])
def save_answer_key():
    data = request.json
    key_name = secure_filename(data.get('name', 'answer_key')) + '.json'
    answers = data.get('answers', [])

    key_path = os.path.join(app.config['ANSWER_KEY_FOLDER'], key_name)
    with open(key_path, 'w') as f:
        json.dump(answers, f)

    return jsonify({'status': 'success', 'filename': key_name})

@app.route('/load_answer_key', methods=['POST'])
def load_answer_key():
    key_name = request.form.get('key_name')
    key_path = os.path.join(app.config['ANSWER_KEY_FOLDER'], key_name)

    if os.path.exists(key_path):
        with open(key_path, 'r') as f:
            answers = json.load(f)
        session['current_answer_key'] = answers
        session['current_answer_key_name'] = key_name
        return jsonify({'status': 'success', 'answers': answers})

    return jsonify({'status': 'error', 'message': 'Answer key not found'}), 404  


@app.route('/preview', methods=['POST'])
def preview():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    params = {
        'threshold': int(request.form.get('threshold', DEFAULT_PARAMS['threshold'])),
            'min_area': int(request.form.get('min_area', DEFAULT_PARAMS['min_area'])),
            'max_area': int(request.form.get('max_area', DEFAULT_PARAMS['max_area'])),
            'questions': int(request.form.get('questions', DEFAULT_PARAMS['questions'])),
            'choices': int(request.form.get('choices', DEFAULT_PARAMS['choices'])),
            'zoom': int(request.form.get('zoom', DEFAULT_PARAMS['zoom'])),
            'width': int(request.form.get('width', DEFAULT_PARAMS['width'])) if 'width' in request.form else None,
            'height': int(request.form.get('height', DEFAULT_PARAMS['height'])) if 'height' in request.form else None,
            'x_offset': int(request.form.get('x_offset', DEFAULT_PARAMS['x_offset'])),
            'y_offset': int(request.form.get('y_offset', DEFAULT_PARAMS['y_offset'])),
            'parts': int(request.form.get('parts', DEFAULT_PARAMS['parts'])),
            'answer_key': session.get('current_answer_key'),
            'answer_key_name': session.get('current_answer_key_name')
    }

    image = cv2.imread(filepath)
    if params['width'] is None:
        params['width'] = image.shape[1]
    if params['height'] is None:
        params['height'] = image.shape[0]

    zoom_factor = params['zoom'] / 100.0
    x, y = params['x_offset'], params['y_offset']
    w, h = int(params['width'] * zoom_factor), int(params['height'] * zoom_factor)

    zoomed_image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
    x_end = min(x + w, zoomed_image.shape[1])
    y_end = min(y + h, zoomed_image.shape[0])
    cropped = zoomed_image[y:y_end, x:x_end]
    #
    # gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, params['threshold'], 255, cv2.THRESH_BINARY_INV)
    #
    # original_path = os.path.join(app.config['RESULT_FOLDER'], 'omr_original.jpg')
    # processed_path = os.path.join(app.config['RESULT_FOLDER'], 'omr_preview.jpg')
    # cv2.imwrite(original_path, cropped)
    # cv2.imwrite(processed_path, thresh)

    # image = cv2.imread(filepath)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, params['threshold'], 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if params['min_area'] < cv2.contourArea(cnt) < params['max_area']]

    valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[1])

    correct_answers = 0
    student_answers = []

    # Column based no of parts
    part_width = image.shape[1] // params['parts']

    for part in range(params['parts']):
        part_contours = [cnt for cnt in valid_contours if part * part_width <= cv2.boundingRect(cnt)[0] <
                         (part + 1) * part_width]
        part_contours = sorted(part_contours, key=lambda c: cv2.boundingRect(c)[1])

        # Process each row (question)
        for i in range(0, min(len(part_contours), params['questions'] * params['choices']), params['choices']):
            question_contours = part_contours[i:i + params['choices']]

            # Sort contours(left to right)
            question_contours = sorted(question_contours, key=lambda c: cv2.boundingRect(c)[0])

            if len(question_contours) != params['choices']:
                continue

            bubble_areas = []
            for j, bubble in enumerate(question_contours):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [bubble], -1, 255, -1)

                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                area = cv2.countNonZero(mask)
                bubble_areas.append((area, j))

            bubble_areas = sorted(bubble_areas, reverse=True)

            # multiple answers or blank answers
            max_area, max_idx = bubble_areas[0]
            second_max_area = bubble_areas[1][0] if len(bubble_areas) > 1 else 0

            if max_area == 0:
                max_idx = -1
            elif max_area > 0 and second_max_area > 0.5 * max_area:
                max_idx = -1
            # Check if answer is correct
            # question_number = (i // params['choices']) + (part * params['questions'] // params['parts'])
            # if question_number < len(params['answer_key']):
            #     is_correct = (max_idx == params['answer_key'][question_number])
            #     color = (0, 255, 0) if is_correct else (0, 0, 255)
            #
            #     correct_answers += 1 if is_correct else 0

            cv2.drawContours(image, [question_contours[max_idx]], -1, (255, 255, 0), 3)
            cv2.putText(image, str(max_idx),(question_contours[max_idx][0][0][0] - 10, question_contours[max_idx][0][0][1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    original_path = os.path.join(app.config['RESULT_FOLDER'], 'omr_original.jpg')
    processed_path = os.path.join(app.config['RESULT_FOLDER'], 'omr_preview.jpg')
    cv2.imwrite(original_path, cropped)
    cv2.imwrite(processed_path, image)
    return jsonify({
        'original_image': original_path,
        'processed_image': processed_path
    })


if __name__ == '__main__':
    app.run(debug=True)
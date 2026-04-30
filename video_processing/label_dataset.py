import base64
import glob
import os
import re
import shutil
import requests
from PIL import Image

API_URL = "http://localhost:1234/v1/chat/completions"

PROMPT = """your job is to detect whats on the registration plate of the vehicle in photos
your input: photos
your output format: [<output>] where <output> is a sequence of SEVEN NUMBERS OR CHARACTERS without whitespaces and NO special characters, these are czech registration plates so on the SECOND PLACE there will always be normal ascii CHARACTER, otherwise its a combination of numbers and characters
if the license plate is not visible in ANY of those photos you are to return just [-]
if the license is very blurry in ALL the images so that you are unsure of what it says, return just [-]
you are NOT to return anything other than the specified output format, no comments, no nothing except output in those "[", "]" brackets
examples of return values (either - or seven characters with second character being a letter, others combination of letters and numbers):
[-]
[1BT0909]
[BZB9828]
[-]"""

def get_image_resolution(image_path):
    try:
        with Image.open(image_path) as img:
            return img.width * img.height
    except Exception as e:
        print(f"ERROR opening {image_path}: {e}")
        return 0

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_top_4_images(folder_path):
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    image_files = []

    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))

    image_files = list(set(image_files))

    if not image_files:
        return []

    image_files.sort(key=get_image_resolution, reverse=True)
    return image_files[:4]

def postproces_plate(plate):
    plate = plate.upper()
    plate = plate.replace("O", "0").replace("Q", "0")

    if len(plate) >= 2:
        plate_list = list(plate)
        second_place = plate_list[1]
        fixes = {
            "8": "B",
            "5": "S",
            "2": "Z",
            "0": "C",
            "7": "T",
            "1": "J",
            "4": "A",
            "P": "P",
            "E": "E",
            "H": "H",
            "K": "K",
            "L": "L",
            "M": "M",
        }

        if second_place in fixes:
            plate_list[1] = fixes[second_place]

        plate = "".join(plate_list)

    return plate

def save_to_output(plate, source_folder, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    target_folder = os.path.join(output_dir, plate)

    if os.path.exists(target_folder):
        #print(f"------------  Identic vehicle found (plate: {plate}) ------------")
        pass
    else:
        os.makedirs(target_folder)

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    all_images = []
    for ext in extensions:
        all_images.extend(glob.glob(os.path.join(source_folder, ext)))
        all_images.extend(glob.glob(os.path.join(source_folder, ext.upper())))
    all_images = list(set(all_images))

    for img_path in all_images:
        img_name = os.path.basename(img_path)
        target_path = os.path.join(target_folder, img_name)

        counter = 1
        base, ext = os.path.splitext(img_name)
        while os.path.exists(target_path):
            target_path = os.path.join(target_folder, f"{base}_{counter}{ext}")
            counter += 1

        shutil.copy2(img_path, target_path)

    #print(f"Copied {len(all_images)} images to '{target_folder}'")

def process_vehicle_folder(folder_path, output_dir):
    """Processes a single vehicle folder, sends to API, and saves the answer."""
    folder_name = os.path.basename(folder_path)
    images_for_api = get_top_4_images(folder_path)

    if not images_for_api:
        return

    print(f"Evaluating car in folder '{folder_name}'")

    content = [{"type": "text", "text": PROMPT}]

    for img_path in images_for_api:
        base64_img = encode_image_to_base64(img_path)
        ext = img_path.split(".")[-1].lower()
        mime_type = f"image/{ext if ext != 'jpg' else 'jpeg'}"

        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_img}"},
        })

    payload = {
        "model": "local-model",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 20,
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()

        reply = data["choices"][0]["message"]["content"].strip()
        match = re.search(r"\[(.*?)\]", reply)

        if match:
            raw_result = match.group(1).strip()
            if raw_result == "-":
                #print(f"PLATE NOT RECOGNIZED (for car in folder '{folder_name}')")
                pass
            else:
                clean_result = postproces_plate(raw_result)
                #print(f"FOUND: {clean_result}")
                save_to_output(clean_result, folder_path, output_dir)
        else:
            print(f"INVALID MODEL ANSWER: {reply}")

    except Exception as e:
        print(f"ERROR: while processing car in folder '{folder_name}': {e}")
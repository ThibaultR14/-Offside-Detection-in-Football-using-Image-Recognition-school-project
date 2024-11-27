import cv2
import numpy as np
from sklearn.cluster import KMeans
from inference_sdk import InferenceHTTPClient
import json

#  Image loading
image_path = "images//Off1.jpg"  #  <--- Change Image here
image = cv2.imread(image_path)

#  Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Zt9QyvjuGZJhKZBLUpAj"
)

# Text
points = []
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_scale2 = 0.7
font_scale3 = 0.8
color = (255, 255, 255)
thickness = 2

def get_text_position(image, text, font_scale, thickness):
    text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness)[0]
    y_position = image.shape[0] - text_height - 20
    return (image.shape[1] // 2 - text_width // 2, y_position)


text1 = 'Select 2 points for horizontal line and 2 for vertical line'
text2 = 'Select in order referee -> attacker -> defender -> goalkeeper'
text3 = 'Data is save, now go on Part2_OffsideProgram.py, press q'

# Function
def detect_dominant_color(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y + h, x:x + w]
    roi = roi.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1, random_state=0).fit(roi)
    return tuple(np.int64(kmeans.cluster_centers_[0]))


def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Selection Zone", image)
        if len(points) == 4:  # 4 points définis
            print("Selected points:", points)
            cv2.destroyAllWindows()

def select_references(image, players):
    references = {"Referee": None, "Attacker": None, "Defender": None, "Goalkeeper": None}
    labels = ["Referee", "Attacker", "Defender", "Goalkeeper"]
    current_label = 0

    def click_event(event, x, y, flags, param):
        nonlocal current_label
        if event == cv2.EVENT_LBUTTONDOWN:
            for player in players:
                px, py, w, h = player
                if px < x < px + w and py < y < py + h:  # Si clic sur un joueur
                    bbox = (px, py, w, h)
                    color = detect_dominant_color(image, bbox)
                    if labels[current_label] == "Goalkeeper" and references["Goalkeeper"] is not None:
                        return
                    references[labels[current_label]] = color
                    print(f"{labels[current_label]} selected: {color}")
                    current_label += 1
                    if current_label == len(labels):  # Si toutes les références sont sélectionnées
                        cv2.destroyAllWindows()
                    break

    print("Cliquez sur les joueurs pour sélectionner : Referee | -> | Attacker | -> | Defender | -> | Goalkeeper.")

    temp_image = image.copy()
    for player in players:
        x, y, w, h = player
        cv2.rectangle(temp_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    xt2, yt2 = get_text_position(temp_image, text2, font_scale2, thickness)
    (text_width, text_height), _ = cv2.getTextSize(text2, font, font_scale2,
                                                   thickness)
    cv2.rectangle(temp_image, (xt2 - 10, yt2 - 10), (xt2 + text_width + 10, yt2 + text_height + 10), (0, 0, 0), -1)
    cv2.putText(temp_image, text2, (xt2, yt2 + text_height + 5), font, font_scale2, color,
                thickness, cv2.LINE_AA)
    cv2.imshow("Select players class", temp_image)
    cv2.setMouseCallback("Select players class", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return references

# Data
def save_data(players_with_classes, lines):
    data = {
        "players": players_with_classes,
        "lines": lines
    }
    with open("players_and_lines.json", "w") as f:
        json.dump(data, f)

def load_data():
    with open("players_and_lines.json", "r") as f:
        data = json.load(f)
    return data



xt1, yt1 = get_text_position(image, text1, font_scale, thickness)
(text_width, text_height), _ = cv2.getTextSize(text1, font, font_scale, thickness)
cv2.rectangle(image, (xt1 - 10, yt1 - 10), (xt1 + text_width + 10, yt1 + text_height + 10), (0, 0, 0), -1)
cv2.putText(image, text1, (xt1, yt1 + text_height + 5), font, font_scale, color, thickness, cv2.LINE_AA)

cv2.imshow("Selection Zone", image)
cv2.setMouseCallback("Selection Zone", click_event)
cv2.waitKey(0)

if len(points) != 4:
    raise ValueError("Select 4 points: 2 for the upper horizontal line of the pitch and 2 for the vertical line of the pitch")

x1, y1 = points[0]
x2, y2 = points[1]

if x2 != x1:
    m_h = (y2 - y1) / (x2 - x1)
    b_h = y1 - m_h * x1
    y_left = int(m_h * 0 + b_h)
    y_right = int(m_h * image.shape[1] + b_h)
else:
    cv2.line(image, (x1, 0), (x1, image.shape[0]), (0, 255, 0), 2)

x3, y3 = points[2]
x4, y4 = points[3]

if x4 != x3:
    m_v = (y4 - y3) / (x4 - x3)
    b_v = y3 - m_v * x3
    x_top = int((0 - b_v) / m_v)
    x_bottom = int((image.shape[0] - b_v) / m_v)
    cv2.line(image, (x_top, 0), (x_bottom, image.shape[0]), (0, 255, 0), 2)
else:
    cv2.line(image, (0, y3), (image.shape[1], y3), (0, 255, 0), 2)

result = CLIENT.infer(image_path, model_id="football-players-detection-3zvbc/12")

players = []
for prediction in result['predictions']:
    if prediction['class'] in ['player', 'goalkeeper']:  # Traiter les deux classes de la même manière
        x = int(prediction['x'] - prediction['width'] / 2)
        y = int(prediction['y'] - prediction['height'] / 2)
        w = int(prediction['width'])
        h = int(prediction['height'])
        players.append([x, y, w, h])

references = select_references(image, players)

players_with_classes = []

for player in players:
    x, y, w, h = player
    dominant_color = detect_dominant_color(image, (x, y, w, h))

    if references["Goalkeeper"] is not None and np.array_equal(dominant_color, references["Goalkeeper"]):
        players_with_classes.append([x + w // 2, y + h // 2, "Goalkeeper"])
    else:
        distances = {label: np.linalg.norm(np.array(dominant_color) - np.array(color))
                     for label, color in references.items() if color is not None and label != "Goalkeeper"}

        if distances:
            closest_label = min(distances, key=distances.get)
            players_with_classes.append([x + w // 2, y + h // 2, closest_label])

black_image = np.zeros_like(image)

for player in players_with_classes:
    px, py, label = player
    if label == "Attacker":
        color = (0, 255, 0)
    elif label == "Defender":
        color = (255, 0, 0)
    elif label == "Referee":
        color = (0, 0, 255)
    elif label == "Goalkeeper":
        color = (255, 255, 0)

    cv2.circle(black_image, (px, py), 10, color, -1)
    cv2.putText(black_image, label.capitalize(), (px + 15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if x2 != x1:
    cv2.line(black_image, (0, y_left), (image.shape[1], y_right), (0, 255, 0), 2)
else:
    cv2.line(black_image, (x1, 0), (x1, image.shape[0]), (0, 255, 0), 2)

if x4 != x3:
    cv2.line(black_image, (x_top, 0), (x_bottom, image.shape[0]), (0, 255, 0), 2)
else:
    cv2.line(black_image, (0, y3), (image.shape[1], y3), (0, 255, 0), 2)

save_data(players_with_classes, {"horizontal_line": (x1, y1, x2, y2), "vertical_line": (x3, y3, x4, y4)})

xt3, yt3 = get_text_position(black_image, text3, font_scale3, thickness)
(text_width, text_height), _ = cv2.getTextSize(text3, font, font_scale3, thickness)
cv2.rectangle(black_image, (xt3 - 10, yt3 - 10), (xt3 + text_width + 10, yt3 + text_height + 10), (20, 20, 20), -1)
cv2.putText(black_image, text3, (xt3, yt3 + text_height + 5), font, font_scale3, (255,255,255), thickness, cv2.LINE_AA)

cv2.imshow("Lines & players black background", black_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

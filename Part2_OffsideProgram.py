import cv2
import numpy as np
import json

# Data
with open('players_and_lines.json', 'r') as file:
    data = json.load(file)

players = data["players"]
lines = data["lines"]

image = np.zeros((500, 1200, 3), dtype=np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_scale2 = 2
color = (255, 255, 255)
thickness = 2

text = 'Select a receiver'
(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)





def get_text_position(image, text, font_scale, thickness):
    text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness)[0]
    y_position = image.shape[0] - text_height - 20
    return (image.shape[1] // 2 - text_width // 2, y_position)

def line_intersection(line1, line2):
    """Finds the intersection of two lines defined by (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Lines are parallel
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return int(px), int(py)

def extend_line(x1, y1, x2, y2, width, height):
    if x1 == x2:
        return [(x1, 0), (x2, height)]
    elif y1 == y2:
        return [(0, y1), (width, y1)]
    else:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        y_at_xw = int(m * width + b)
        y_at_x0 = int(b)
        return [(0, y_at_x0), (width, y_at_xw)]

h_line = extend_line(*lines["horizontal_line"], image.shape[1], image.shape[0])
v_line = extend_line(*lines["vertical_line"], image.shape[1], image.shape[0])

vanishing_point = line_intersection(
    (h_line[0][0], h_line[0][1], h_line[1][0], h_line[1][1]),
    (v_line[0][0], v_line[0][1], v_line[1][0], v_line[1][1])
)


cv2.line(image, h_line[0], h_line[1], (0, 255, 0), 2)
cv2.line(image, v_line[0], v_line[1], (0, 255, 0), 2)

receiver = None


def select_receiver(event, x, y, flags, param):
    global receiver
    if event == cv2.EVENT_LBUTTONDOWN:
        for player in players:
            px, py, label = player
            if label == "Attacker" and abs(px - x) < 10 and abs(py - y) < 10:
                receiver = player
                print(f"Receiver selected: {receiver}")
                break
        if receiver:
            cv2.destroyAllWindows()

for player in players:
    px, py, label = player
    color = (0, 255, 0) if label == "Attacker" else (255, 0, 0) if label == "Defender" else (0, 0, 255)
    cv2.circle(image, (px, py), 10, color, -1)
    cv2.putText(image, label.capitalize(), (px + 15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

print("Select a forward player to be the receiver")
xt1, yt1 = get_text_position(image, 'Select a receiver', font_scale2, thickness)
cv2.putText(image, 'Select a receiver', (xt1, yt1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
cv2.imshow("Who is the receiver?", image)
cv2.setMouseCallback("Who is the receiver?", select_receiver)
cv2.waitKey(0)

defenders = [p for p in players if p[2] in ["Defender", "Goalkeeper"]]
defenders_sorted = sorted(defenders, key=lambda p: abs(p[0] - lines["vertical_line"][0]))
second_defender = defenders_sorted[1] if len(defenders_sorted) > 1 else None


if second_defender and receiver:
    result_image = image.copy()
    rec_line = [(receiver[0], receiver[1]), vanishing_point]
    def_line = [(second_defender[0], second_defender[1]), vanishing_point]

    cv2.line(result_image, rec_line[0], rec_line[1], (255, 255, 0), 2)
    cv2.line(result_image, def_line[0], def_line[1], (255, 0, 255), 2)
    rect_start = (xt1 - 10, yt1 - text_height - 10)
    rect_end = (xt1 + text_width + 10, yt1 + baseline + 10)
    cv2.rectangle(result_image, rect_start, rect_end, (0, 0, 0), -1)  # Rectangle cyan


    if receiver:
        # Entourer le joueur receveur en cyan
        cv2.circle(result_image, (receiver[0], receiver[1]), 20, (255, 255, 0), 2)  # Cercle cyan autour du receveur
        cv2.putText(result_image, "Receiver", (receiver[0] + 25, receiver[1] - 10), font, 0.5, (255, 255, 0), 2)

    if rec_line[0][0] < def_line[0][0]:
        xt2, yt2 = get_text_position(result_image, 'OFFSIDE', font_scale2, thickness)
        cv2.putText(result_image, 'OFFSIDE', (xt2, yt2), font, font_scale2, (0, 0, 255), 6, cv2.LINE_AA)
    else:
        xt3, yt3 = get_text_position(result_image, 'NOT OFFSIDE', font_scale2, thickness)
        cv2.putText(result_image, 'NOT OFFSIDE', (xt3, yt3), font, font_scale2, (0, 255, 0), 6, cv2.LINE_AA)


    cv2.imshow("Offside verification", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Receiver or defenders insufficient to check offside.")

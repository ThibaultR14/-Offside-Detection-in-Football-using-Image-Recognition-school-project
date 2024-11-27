# **Offside Detection in Football using Image Recognition**

## **Description**  
This project aims to automate the detection of offside situations in football using image recognition techniques. The system detects players on the field and analyzes their positions relative to the offside line.  

---

## **Motivation**  
The increasing complexity and speed of football matches make it challenging for referees to assess offside situations in real-time. Inspired by the introduction of VAR (Video Assistant Referee), this project seeks to provide an automated solution for accurate and reliable offside detection, assisting referees and analysts.  

---

## **Features**  
- Detection and classification of players (forward, defender, goalkeeper, referee).  
- Determination and visualization of the offside line.  
- Automated analysis to detect offside situations.  
- Visual results to aid interpretation.  

---

## **Technologies Used**  
- **Python**: Main programming language.  
- **OpenCV**: For image processing and manipulation.  
- **Scikit-learn**: Used for clustering algorithms to detect dominant colors. 
- **inference_sdk** : For using pre-trained model from ***Roboflow***

---

## **How to use the code**  



   1. **Install the necessary dependencies with the following command:**  
   ```bash
    pip install opencv-python numpy scikit-learn
   ```
* Place your images in the appropriate folder *"images"*.
* And change image in the 8 line of ***"Part1_DetectionLines&Players.py"***:
```bash
image_path = "images//Off1.jpg"  #  <--- Change Image here
```
* By default, use ***images/Off1.jpg***.

2. **How the code works**

* ***Run the main script:***

```bash
python Part1_SaveData.py
```
* A window will appear displaying the field image:

![Image](/Pic4readme/pic6.jpg)

* ***Click on 4 points to define the field lines:***
* The first 2 points for the ***upper horizontal line***.
* The last 2 points for a ***vertical line***

![Image](/Pic4readme/pic1.png)

* Once the lines are set, select the player classes:

![Image](/Pic4readme/pic2.png)

* Click on the players in the order: ***referee → attacker → defender → goalkeeper***.


* The data will be saved in a file named players_and_lines.json.
* An image with classified players and lines will be displayed. Close the window with q.

![Image](/Pic4readme/pic3.png)


* ***Run the second script:*** Execute the verification script:


```bash
python Part2_OffsideProgram.py
```
* A black window will appear showing the lines and placed players:

![Image](/Pic4readme/pic4.png)

* Click on the attacker you want to designate as the receiver.
* The program will analyze whether the receiver is offside based on the defenders and display the result:
* OFFSIDE if the receiver is offside.
* NOT OFFSIDE otherwise.


* **Result**

![Image](/Pic4readme/pic5.png)

* Close the window after viewing the result.

***Video of the project***
* link : https://streamable.com/1oyoef



---

## **How the Code Works**

***Part 1: Player Selection and Line Setup***
1. **Image and Model Initialization:**

- The program starts by loading an image of the football pitch (Off1.jpg) and initializes a Roboflow model for player detection using the InferenceHTTPClient class.

2. **User Interaction for Line Setup:**

- The user is prompted to select four points on the pitch: two to define a horizontal line and two for a vertical line. These lines represent pitch boundaries and assist in detecting offside situations.
3. **Player Detection:**

- Players are detected in the image using the Roboflow model. Detected players are stored as bounding boxes, each representing a player's position and size.
4. **Player Classification by Roles:**

- The user is guided to click on specific players to classify them as "Referee," "Attacker," "Defender," or "Goalkeeper." The program uses the dominant color of the selected player's region to associate their role.
5. **Player Data and Line Saving:**

- The classified players, along with the pitch lines, are saved into a JSON file (players_and_lines.json) for further processing in Part 2.
6. Visual Representation:

- A new image is created to highlight the players' positions and the defined pitch lines. Colors indicate player roles (e.g., attackers in green, defenders in blue, etc.), and the horizontal and vertical lines are drawn for reference.

***Part 2: Offside Verification***
1. **Data Loading:**

- The JSON file from Part 1 is loaded to retrieve the players' positions and pitch lines.
2. **Vanishing Point Calculation:**

- The program calculates the intersection (vanishing point) of the horizontal and vertical lines to establish perspective.
3. **Receiver Selection:**

- The user selects an attacking player who will be the potential receiver of the ball. The program identifies the selected player based on their proximity to the mouse click.
4. **Second-to-Last Defender Identification:**

- Among the defenders and the goalkeeper, the program identifies the second-to-last defender. This player is critical in determining the offside position.
5. **Offside Decision:**

- Lines are drawn from the receiver and the second-to-last defender to the vanishing point. If the receiver is closer to the goal line than the second-to-last defender, the player is deemed offside. Otherwise, they are not offside.
6. **Result Visualization:**

- The program visually highlights the receiver and displays an "OFFSIDE" or "NOT OFFSIDE" message on the screen, depending on the result.
7. **Program End:**

- If sufficient data is not available (e.g., no receiver or defenders are selected), the program notifies the user and terminates.

---

## **Performance Metrics**

***The model achieves the following performance metrics:***

- mAP (Mean Average Precision): 91.2%
- Precision: 92.8% 
- Recall: 85.3%
- For player detection, the estimated performance is approximately 90% accuracy.

---

## **Contributors**
* **Thibault RABBE**: KNU ID : 202400518 : Development, integration, and project documentation.

* **Roboflow:** Provider of the pre-trained model for player detection.
* **OpenCV:** Used for perspective mapping with the solvePnP function.

---

## **References**
* Pre-trained Player Detection Model: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/m
* OpenCV Documentation - solvePnP: https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
* Video Sources for Match Footage:
* https://www.youtube.com/watch?v=9gyv2xh7qQw
* https://www.youtube.com/watch?v=ZeJlh0vhcFc
* https://www.youtube.com/watch?v=2mPQiz_Ig8U

---

## **Futur works**

* Improve player detection with more efficient models like YOLO or Faster R-CNN.

* Real-time tracking of players to avoid re-detection in each frame.

* Use multiple cameras for more robust detection and multiple perspectives.

* Optimize system performance for real-time use with GPUs or parallelization.
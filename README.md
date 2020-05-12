# Basketball-Shot-Detectection

Major improvements data extraction in sports enables sport analyst to study and output an evaluation report of an athletes capabilities. Major teams and even players themselves have acted quickly to these turn of events and are now investing in the
idea of data driven decisions to win games.

For my final year project I have decided to create an computer vision application based on the sport of Basketball. The following application will take in a recorded video of a player shooting a basktball as its input. Using computer vision methods in python I aim to display player position on a 2D court and display total count of make and missed shots on a score board.

#### Project Explanation/ Demo: <a href="https://www.youtube.com/watch?v=aW3IlB3nBoI">https://www.youtube.com/watch?v=aW3IlB3nBoI</a>
#### Project Poster: <a href="https://github.com/Esedicol/BasketballPlayerDetectection-BABPD/blob/master/FYP_DOCUMENTS/POSTER.pdf">/FYP_DOCUMENTS/POSTER.pdf</a>
#### Project Report: <a href="https://github.com/Esedicol/BasketballPlayerDetectection-BABPD/blob/master/FYP_DOCUMENTS/KCOMP_20072377_FinalReport.pdf">/FYP_DOCUMENTS/KCOMP_20072377_FinalReport.pdf</a>


## Ball Detection using OpenCV
![Demo](RESULTS/DEMO.gif)

## Ball Detection using OpenCV
- frame masked using lower and upper hsv colour range 
- countour detection and labelling usingOpenCV an
![Demo](RESULTS/BALL_D.gif)

## Player Detection using YOLO 
- detection player using YOLOv3
- label using OpenCV

<img src="RESULTS/PLAYER_D.gif" height="300">

## 3D player position to 2D player position 
- extract points using hough transform and line intersection methods
- warped 3D points onto coresponding 2D points
- detection player using colour range masking
![Demo](RESULTS/POSITION_D.gif)

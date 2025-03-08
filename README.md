# ChessBot: Vision-Guided Chess Playing with UR10e Robot

This repository contains the implementation of an intelligent chess-playing robot system using a UR10e collaborative robot arm. The system integrates computer vision, motion planning, and AI to enable a robot to play chess against human opponents. Video link (https://drive.google.com/drive/u/2/folders/1obFIqRed-Po3y0oRw5mMarEn5gsubKf2)

## Overview

ChessBot employs a comprehensive approach to robotic chess playing:

1. **Computer Vision** - Detects the chessboard and tracks piece movements using OpenCV
2. **Motion Planning** - Uses ROS and MoveIt! to navigate the UR10e robot arm safely
3. **AI Integration** - Leverages Stockfish chess engine for strategic gameplay decisions

The system captures images of the chessboard before and after a human player makes a move, identifies the move through image processing techniques, and instructs the robot to make a strategic countermove based on Stockfish AI analysis.

## Features

- **UR10e Integration**: Utilizes the 6-DoF UR10e collaborative robot with 1.3m reach and 10kg payload capacity
- **ROS Framework**: Implements standardized robotic control through ROS and MoveIt!
- **Advanced Path Planning**: Employs OMPL with RRT algorithm and CHOMP as backup
- **Image Difference Algorithm**: Detects chess moves through image subtraction and thresholding
- **Chessboard Coordinate Mapping**: Converts pixel locations to real-world coordinates
- **Stockfish AI Integration**: Provides strategic gameplay decisions
- **3D Printed Hardware**: Custom camera mount and gripper claws for chess piece manipulation

## System Architecture

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│   Computer     │     │     ROS &      │     │    UR10e       │
│   Vision       │────>│    MoveIt!     │────>│  Robot Arm     │
│  (OpenCV)      │     │  (Path Planning)│     │                │
└────────────────┘     └────────────────┘     └────────────────┘
        │                      ▲                      │
        │                      │                      │
        ▼                      │                      ▼
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│   Image        │     │   Stockfish    │     │   Chess        │
│   Processing   │────>│   Chess AI     │────>│   Pieces       │
│                │     │                │     │                │
└────────────────┘     └────────────────┘     └────────────────┘
```

## Key Components

### 1. Computer Vision System
- Captures chessboard images before and after human moves
- Implements image difference algorithm to detect changes
- Applies thresholding and morphological operations for noise reduction
- Maintains a CSV database of chess square positions and piece status

### 2. Motion Planning
- Uses OMPL (Open Motion Planning Library) with RRT (Rapidly-exploring Random Tree) algorithm
- Implements CHOMP (Covariant Hamiltonian Optimization for Motion Planning) as backup
- Verifies paths for self-collision and workspace violations
- Ensures safe and efficient robotic movement

### 3. Chess AI Integration
- Communicates with Stockfish chess engine
- Translates detected human moves into chess notation
- Receives AI-suggested moves for the robot
- Implements strategic gameplay decisions

## Technical Details

### Image Processing Pipeline
1. Capture reference image of empty/initial chessboard
2. Detect chessboard corners using OpenCV
3. Create 8x8 grid mapping physical squares to image coordinates
4. For each move:
   - Capture pre-move and post-move images
   - Convert to grayscale and calculate image difference
   - Apply thresholding and morphological operations
   - Detect significant changes and map to chess squares
   - Update piece position database

### Robot Control
- MoveIt! Commander provides high-level robot control interface
- Master controller verifies paths before execution
- Custom path planning strategies avoid collisions and optimize movements
- Real-world coordinate transformation maps camera space to robot space

### 3D Printed Components
- Custom camera mount compatible with most webcams
- Specialized gripper with extended claws for diverse chess piece manipulation

## Requirements

- ROS (Robot Operating System)
- Universal Robots UR10e robot arm
- OpenCV for Python
- MoveIt! motion planning framework
- Stockfish chess engine
- Python 3.x with NumPy, Pandas
- Camera/webcam (MI webcam recommended)

## Setup and Usage

1. Clone this repository:
   ```
   git clone https://github.com/Kathanpatel7/chessbot.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure your UR10e robot and ROS environment according to the documentation

4. Run the main chess application:
   ```
   python main.py
   ```

## Future Enhancements

- Improved piece recognition for different chess set styles
- Support for special moves (castling, en passant, promotion)
- Voice control and interactive gameplay features
- Machine learning model for piece detection in varied lighting conditions
- Mobile app interface for game monitoring and control

## Contributors

- Kathan Patel ([@Kathanpatel7](https://github.com/Kathanpatel7))
- Masoom Lalani

## Acknowledgements

Special thanks to Vishal Vaidya for guidance and mentorship throughout this project, and to Dr. Sharma for proofreading support. This project was developed as part of the Bachelor of Technology degree requirements in Instrumentation and Control Engineering at Nirma University, Ahmedabad.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
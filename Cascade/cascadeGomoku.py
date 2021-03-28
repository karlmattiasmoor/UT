import numpy as np
import cv2
from math import sqrt
from os.path import exists

# variables

boardSize = 15
frameSize = None

# important file locations:

testimg = 'Cascade/WIN_20201228_18_49_56_Pro.jpg'
blackCascadeFile = "Cascade/blackCascade.xml"
whiteCascadeFile = "Cascade/whiteCascade.xml"
emptyCascadeFile = "Cascade/emptyCascade.xml"

mapx = None
mapy = None


def closeWindow(win="video"):
    cv2.destroyWindow(win)
    for i in range(4):
        cv2.waitKey(1)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def readImage():
    image = cv2.imread(testimg)
    image = image_resize(image, height=600)
    return image


empty_cascade = cv2.CascadeClassifier(emptyCascadeFile)
black_cascade = cv2.CascadeClassifier(blackCascadeFile)
white_cascade = cv2.CascadeClassifier(whiteCascadeFile)


def readBoard(image):


    output = np.zeros((boardSize, boardSize), dtype="uint8")
    imgCorners = None

    emptyRectangles = empty_cascade.detectMultiScale(image, 1.035, 2)
    # a black stone:
    blackRectangles = black_cascade.detectMultiScale(image, 1.1, 1)
    # or a white stone:
    whiteRectangles = white_cascade.detectMultiScale(image, 1.15, 1)
    
    empties = []
    blacks = []
    whites = []
    for (ex, wy, w, h) in emptyRectangles:
        x = ex + w / 2.0
        y = wy + w / 2.0
        empties.append([x, y])
    for (ex, wy, w, h) in blackRectangles:
        x = ex + w / 2.0
        y = wy + w / 2.0
        blacks.append([x, y])
    for (ex, wy, w, h) in whiteRectangles:
        x = ex + w / 2.0
        y = wy + w / 2.0
        whites.append([x, y])

    for c in empties:
        cv2.circle(image,
                   (int(round(c[0])), int(round(c[1]))),
                   3,
                   (0, 255, 255),
                   -1)
    for c in blacks:
        cv2.circle(image,
                   (int(round(c[0])), int(round(c[1]))),
                   3,
                   (0, 255, 0),
                   -1)
    for c in whites:
        cv2.circle(image,
                   (int(round(c[0])), int(round(c[1]))),
                   3,
                   (0, 0, 255),
                   -1)
    cv2.imshow("dots", image)

    group = findGroup(empties + blacks + whites)
    if group is None:
        return output, imgCorners
    hull = cv2.convexHull(np.array(group, dtype="int32"))
    epsilon = 0.001*cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    imgCorners = fourCorners(approx)
    if imgCorners is not None and len(imgCorners) > 3:
        flatCorners = np.array([[0, 0],
                                [boardSize - 1, 0],
                                [boardSize - 1, boardSize - 1],
                                [0, boardSize - 1]],
                               dtype="float32")
        persp = cv2.getPerspectiveTransform(imgCorners, flatCorners)
        if len(blacks) > 0:
            blacks = np.array(blacks, dtype="float32")
            blacks = np.array([blacks])
            blacksFlat = cv2.perspectiveTransform(blacks, persp)
            for i in blacksFlat[0]:
                x = int(round(i[0]))
                y = int(round(i[1]))
                if x >= 0 and x < boardSize and y >= 0 and y < boardSize:
                    output[x][y] = 1
        if len(whites) > 0:
            whites = np.array(whites, dtype="float32")
            whites = np.array([whites])
            whitesFlat = cv2.perspectiveTransform(whites, persp)
            for i in whitesFlat[0]:
                x = int(round(i[0]))
                y = int(round(i[1]))
                if x >= 0 and x < boardSize and y >= 0 and y < boardSize:
                    output[x][y] = 2

    return output, imgCorners


def findGroupMembers(maxDistance, i, distances, group):
    for j in range(len(group)):
        if group[j]:
            pass
        elif distances[i][j] < maxDistance:
            group[j] = True
            findGroupMembers(maxDistance, j, distances, group)

def findGroup(spots):
    length = len(spots)
    distances = np.zeros((length, length), dtype="float32")
    distanceList = []
    for i in range(length):
        for j in range(length):
            d = sqrt((spots[i][0] - spots[j][0])**2 + (spots[i][1] - spots[j][1])**2)
            distances[i][j] = d
            if d > 0:
                distanceList.append(d)

    distanceList.sort()
    numDistances = int((boardSize - 1)**2 * 1.8)
    maxDistance = np.mean(distanceList[0:numDistances]) * 1.75

    # find a big enough group
    minGroup = int(boardSize**2 * 0.6)
    group = np.zeros((length), dtype="bool_")
    for i in range(length):
        findGroupMembers(maxDistance, i, distances, group)
        if group.sum() >= minGroup:
            outPoints = []
            for k in range(length):
                if group[k]:
                    outPoints.append(spots[k])
            return outPoints
        else:
            group = np.zeros((length), dtype="bool_")

def sortPoints(box):
    rect = np.zeros((4, 2), dtype = "float32")

    s = box.sum(axis = 1)
    rect[0] = box[np.argmin(s)]
    rect[2] = box[np.argmax(s)]

    diff = np.diff(box, axis = 1)
    rect[1] = box[np.argmin(diff)]
    rect[3] = box[np.argmax(diff)]

    return rect

def fourCorners(hull):
    length = len(hull)

    if length < 4:
        return []
    
    allLines = []
    for i in range(length):
        if i == (length - 1):
            line = [[hull[i][0][0], hull[i][0][1]], [hull[0][0][0], hull[0][0][1]]]
        else:
            line = [[hull[i][0][0], hull[i][0][1]], [hull[i + 1][0][0], hull[i + 1][0][1]]]
        d = sqrt((line[0][0] - line[1][0])**2 + (line[0][1] - line[1][1])**2)
        allLines.append([line, d])

    allLines.sort(key=lambda x: x[1], reverse=True)
    lines = []
    for i in range(4):
        lines.append(allLines[i][0])

    equations = []
    for i in lines:
        x_coords, y_coords = zip(*i)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = np.linalg.lstsq(A, y_coords)[0]
        equations.append([m, c])
        
    intersections = []
    for i in equations:
        for j in equations:
            if i[0] == j[0]:
                pass
            else:
                a = np.array([[i[0] * -1, 1], [j[0] * -1, 1]])
                b = np.array([i[1], j[1]])
                solution = np.linalg.solve(a, b)
                if solution[0] > 0 and solution[1] > 0 and solution[0] < frameSize[0] and solution[1] < frameSize[1]:
                    intersections.append([solution[0], solution[1]])

    intersections.sort()
    
    if len(intersections) > 6:
        output = [intersections[0],
                  intersections[2],
                  intersections[4],
                  intersections[6]]
        box = sortPoints(np.array(output, dtype="float32"))
        return box
    else:
        return []

def blankBoard(boardBlockSize):
    yellow = [75, 215, 255]
    black = [0, 0, 0]
    white = [255, 255, 255]
    halfBoardBlock = int(round((boardBlockSize / 2.0)))
    boardSide = boardBlockSize * boardSize
    blankBoard = np.zeros((boardSide, boardSide, 3),
                     dtype="uint8")
    cv2.rectangle(blankBoard, (0, 0), (boardSide, boardSide), yellow, -1)
    for i in range(boardSize):
        spot = i * boardBlockSize + halfBoardBlock
        cv2.line(blankBoard,
                 (spot, halfBoardBlock),
                 (spot, boardSide - halfBoardBlock),
                 black,
                 2)
        cv2.line(blankBoard,
                 (halfBoardBlock, spot),
                 (boardSide - halfBoardBlock, spot),
                 black,
                 2)
    if boardSize == 19:
        spots = [[3, 3], [8, 3], [15, 3],
                 [3, 8], [8, 8], [15, 8],
                 [3, 15], [8, 15], [15, 15]]
    else:
        spots = []

    for s in spots:
        cv2.circle(blankBoard,
                   (s[0] * boardBlockSize + halfBoardBlock,
                    s[1] * boardBlockSize + halfBoardBlock),
                   int(boardBlockSize * .15),
                   black,
                   -1)

    return blankBoard

def drawBoard(board, size=(500, 500)):
    black = [0, 0, 0]
    white = [255, 255, 255]
    
    boardBlockSize = 100
    halfBoardBlock = int(round(boardBlockSize / 2.0))
    output =  blankBoard(100)
    (w, h) = board.shape
    for x in range(w):
        for y in range(h):
            if board[x][y] == 1:
                cv2.circle(output,
                           ((x * boardBlockSize) + halfBoardBlock,
                            (y * boardBlockSize) + halfBoardBlock),
                           int(boardBlockSize / 2),
                           black,
                           -1)
            elif board[x][y] == 2:
                cv2.circle(output,
                           ((x * boardBlockSize) + halfBoardBlock,
                            (y * boardBlockSize) + halfBoardBlock),
                           int(boardBlockSize / 2),
                           white,
                           -1)
    output = cv2.resize(output, size, output, 0, 0, cv2.INTER_AREA)
    return output

def watchBoard():
    global frameSize

    imgCorners = None

    img = readImage()
    (h, w, d) = img.shape
    frameSize = (w, h)
    imageSize = (int(round(w / 2.0)), int(round(h / 2.0)))
    cv2.imshow("camera", cv2.resize(img, imageSize, 0, 0, cv2.INTER_AREA))
    roi = np.zeros((h, w), dtype="uint8")
    cv2.rectangle(roi, (0, 0), (w, h), 1, -1)
    
    while cv2.waitKey(1) == -1:
        new = readImage()
        bkg = readImage()

        board, imgCorners = readBoard(bkg)
        cv2.imshow("board", drawBoard(board))
        if imgCorners is not None and len(imgCorners) > 3: 
            roi = np.zeros((h, w), dtype="uint8")
            cv2.fillConvexPoly(roi, np.array(imgCorners, dtype="int32"), 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
            roi = cv2.dilate(roi, kernel)

        image = new.copy()
        if imgCorners is not None:
            for c in imgCorners:
                cv2.circle(image,
                           (int(round(c[0])), int(round(c[1]))),
                           6,
                           (0, 0, 255),
                           -1)
        cv2.imshow("camera", cv2.resize(image, imageSize, 0, 0, cv2.INTER_AREA))

    closeWindow("camera")
    closeWindow("board")
watchBoard()
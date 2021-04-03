import numpy as np
import cv2

boardSize = 15

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
    if boardSize == 15:
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
            if board[x][y] == 0:
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
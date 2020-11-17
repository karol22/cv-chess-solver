import cv2
import chess
import torch
import chess.svg
import chess.engine

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from time import sleep
from PIL import Image
from torchvision.transforms import ToTensor


# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool4 = nn.MaxPool2d(4,4)
        # First conv layers
        self.conv1 = nn.Conv2d(3, 64, 7, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256,512, 7, padding=3)
        #self.conv5 = nn.Conv2d(12,6, 3, padding=1)
        #self.conv6 = nn.Conv2d(6,3, 1, padding=0)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 14)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool2(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool4(x)
        x = F.relu(self.conv4(x))
        
        x = self.pool4(x)
        #x = F.relu(self.conv5(x))
        #x = F.relu(self.conv6(x))
        x = x.view(x.shape[0], -1)
        # add dropout layer
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# initialize the NN
model = ConvAutoencoder()
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
mapping = {
    'alfil_azul': 0,
    'alfil_rojo': 1,
    'caballo_azul': 2,
    'caballo_rojo': 3,
    'fondo_blanco': 4,
    'fondo_negro': 5,
    'peon_azul': 6,
    'peon_rojo': 7,
    'reina_azul': 8,
    'reina_rojo': 9,
    'rey_azul': 10,
    'rey_rojo': 11,
    'torre_azul': 12,
    'torre_rojo': 13
}
rev_mapping = {v:k for k,v in mapping.items()}


def idx2piece(idx):
    return get_piece(rev_mapping[idx.item()])


def get_piece(piece):
    return {
        'alfil_azul': 'b',
        'alfil_rojo': 'B',
        'caballo_azul': 'n',            
        'caballo_rojo': 'N',
        'peon_azul': 'p',
        'peon_rojo': 'P',
        'reina_azul': 'q',
        'reina_rojo': 'Q',
        'rey_azul': 'K',
        'rey_rojo': 'k',
        'torre_azul': 'r',
        'torre_rojo': 'R'
    }.get(piece, '.')

    
def from_board_to_fne(board):
    fne_board = []
    get_piece_ = lambda piece: {".": False}.get(piece, piece)
    # 'for loop' until the shape is defined
    for row in board.splitlines():
        fne_row = ""
        empty_spaces = 0
        for piece in row:
            piece = get_piece_(piece)
            if piece:
                if empty_spaces:
                    fne_row += str(empty_spaces)
                    fne_row += piece
                else:
                    fne_row += piece

                empty_spaces = 0
            else:
                empty_spaces += 1

        if empty_spaces:
            fne_row += str(empty_spaces)

        fne_board.append(fne_row)

    return "/".join(fne_board)[1:]
    


vid = cv2.VideoCapture(0)
if(vid.isOpened() == False):
    print('Error opening video stream or file')

while(vid.isOpened()):
    ret, img = vid.read()
    if ret == True:
        img = cv2.imread('original.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rows,cols,ch = img.shape
        pts1 = np.float32([[92,152],[737, 142],[57, 764],[780, 770]])
        pts2 = np.float32([[0,0],[512,0],[0,512],[512,512]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(512,512))

        squares = np.array([
            dst[64*i:64*i + 64, 64*j:64*j + 64]
            for i in range(8)
            for j in range(8)
        ])
        squares = squares.transpose((0, 3, 1, 2))
        x = torch.from_numpy(squares).float()
        y = model(x)
        y = model(x)
        y = torch.max(y, 1)

        board = [[None for j in range(8)] for j in range(8)]
        for i, piece in enumerate(map(idx2piece,y.indices)):
            board[i % 8][i // 8] = piece
            
        board = '\n'.join(str(''.join(row)) for row in board)
        board_str = '\n' + board
        board_fen = from_board_to_fne(board_str)

        limit = chess.engine.Limit(time=5.0)

        board_white = chess.Board(f'{board_fen} w')
        board_black = chess.Board(f'{board_fen} b')

        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        x = engine.play(board_white, limit)
        
        print('Best move for white:', x.move)
        x = engine.play(board_black, limit)
        print('Best move for black:', x.move)

        if(cv2.waitKey(25) & 0xFF == ord('q')):
            break
    else:
        break
vid.release()
